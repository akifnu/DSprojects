export const meta = {
  name: 'reel-job',
  description: 'Drive one reel job to a terminal state: a detached worker runs the mechanical steps, agents verify the style, plan, and review the renders; every review finding is recorded',
  whenToUse: 'After `reel job new` (or `reel job add`) has registered the inputs; args: {job: "<job id>"}. Run it again with the same id to resume.',
  phases: [
    { title: 'Run', detail: 'reel-runner: `reel job start` + `reel job wait` (ingest, analysis, transcription, editing copies, render + QC, delivery)' },
    { title: 'Verify style', detail: 'sample-analyst checks the measured style against the sample and answers the need with `reel job done`' },
    { title: 'Plan', detail: 'reel-planner picks the moments, writes plan.json and answers the need with `reel job done`' },
    { title: 'Review', detail: 'qc-reviewers judge the assigned renders in parallel; the merged envelope is recorded with every finding' },
  ],
}

// Python (reelkit/orchestrator.py) owns the job: state, revisions, validation, retry budgets and
// the final verdict. This script only hands each open need to its agent. Commands are built here
// from validated tokens (job id, need id, revision, reel ids, Python-made paths); model-written
// text reaches Python only through files written with the Write tool, never through a shell line.

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const job = typeof args === 'string' ? args : (args && typeof args === 'object' ? args.job : undefined)
if (typeof job !== 'string' || !ID_RE.test(job) || job === 'latest') {
  throw new Error('args.job must be the job id printed by `reel job new` (letters, digits, ".", "_", "-", max 64; not "latest")')
}

const MAX_STEPS = 40       // runner turns (start + wait until the worker settles) before handing back, resumable
const MAX_SAME_NEED = 3    // hand-offs of one open need to its agent before stopping (Python caps rejections too)
const MAX_FAILURES = 3     // consecutive failed turns (runner died, worker died / failed without a terminal state)
const REVIEW_CHUNK = 8     // reels per qc-reviewer
const WAIT_S = 500         // per `reel job wait`; the runner's Bash call allows 600 s
const MAX_WAITS = 20       // waits per runner turn (~2.8 h) before it reports "still running" and a new turn starts

const CD = 'cd reel-pipeline && '
const tailed = (cmd, tag, n) => `${CD}{ ${cmd}; echo "${tag}=$?"; } 2>&1 | tail -n ${n}`
const START = tailed(`./reel job start ${job}`, 'reel-start-exit', 20)
const WAIT = `${CD}{ ./reel job wait ${job} --timeout ${WAIT_S} >/dev/null; echo "reel-wait-exit=$?"; } 2>&1 | tail -n 5`
// `reel job status` reduced to what the workflow needs (a 50-reel review need is too big to copy);
// the worker log tail is attached when the attempt failed or died.
const BRIEF = [
  'import json,os,sys',
  't=sys.stdin.read()',
  's=json.loads(t) if t.lstrip().startswith("{") else {"status":"unknown","terminal":False,"error":{"cause":"no status JSON: "+t[-400:]}}',
  'n=s.get("needs") if isinstance(s.get("needs"),dict) else None',
  'a=s.get("attempt") or {}',
  'd=s.get("deliver") if isinstance(s.get("deliver"),dict) else None',
  'e=s.get("error")',
  'c=lambda v:v[-600:] if isinstance(v,str) else v',
  'K=("step","need_id","revision","agent","schema_version","plan_generation","result_file","manifest","style","plan","review","review_out","sample","sample_id","count","target_duration","lang","only","repair_round","rounds_left_by_reel","rejections")',
  'L=a.get("log") if a.get("state")=="died" or a.get("exit") not in (0,None) or e else None',
  'print(json.dumps({"id":s.get("id"),"status":s.get("status") or "unknown","terminal":bool(s.get("terminal")),"revision":s.get("revision"),"next":s.get("next"),' +
    '"needs":None if n is None else dict({k:c(n[k]) for k in K if n.get(k) is not None},errors=[str(x)[:400] for x in (n.get("errors") or [])[:10]],assigned_ids=[x.get("id") for x in n.get("assigned") or [] if isinstance(x,dict)]),' +
    '"attempt":{k:a.get(k) for k in ("id","state","stage","exit","log")} if a else None,' +
    '"error":{k:c(e[k]) for k in ("stage","cause","type","transient","returncode","stderr_tail") if e.get(k) is not None} if isinstance(e,dict) else c(e),' +
    '"attention":s.get("attention"),"waiting":s.get("waiting"),"counts":s.get("counts"),"findings_open":len(s.get("findings_open") or []),' +
    '"deliver":None if d is None else dict({k:d.get(k) for k in ("files","review","contact_sheet","manifest","counts","shortfall")},unresolved=len(d.get("unresolved") or [])),' +
    '"log_tail":open(L,"rb").read()[-2500:].decode("utf-8","replace") if L and os.path.isfile(L) else None},indent=1,ensure_ascii=False))',
].join('; ')
const STATUS_CMD = `${CD}./reel job status ${job} | python3 -c '${BRIEF}'`
const NEED_CMD = `${CD}./reel job status ${job} | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("needs"),indent=1,ensure_ascii=False))'`
const assignCmd = (ids) => `${CD}./reel job status ${job} | python3 -c 'import json,sys; n=json.load(sys.stdin).get("needs") or {}; w=set(sys.argv[1:]); ` +
  `print(json.dumps({"need_id":n.get("need_id"),"plan_generation":n.get("plan_generation"),"style":n.get("style"),"plan":n.get("plan"),` +
  `"assigned":[a for a in n.get("assigned") or [] if a.get("id") in w]},indent=1,ensure_ascii=False))' ${ids.join(' ')}`

const SAFE_PATH = /^\/[A-Za-z0-9 ._@+=,:~\/-]*$/   // single-quotable as is: no quote, $, backtick, backslash
const safePath = (p) => typeof p === 'string' && SAFE_PATH.test(p) && !p.split('/').includes('..')
const dirOf = (p) => p.slice(0, p.lastIndexOf('/'))
const str = (v, n) => (typeof v === 'string' ? v : v == null ? '' : String(v)).slice(0, n)
const revFlag = (n) => (Number.isInteger(n.revision) ? ` --revision ${n.revision}` : '')

const FINDING = {
  type: 'object',
  properties: {
    code: { type: 'string' },
    severity: { type: 'string', enum: ['blocking', 'minor'] },
    what: { type: 'string' },
    how: { type: 'string' },
  },
  required: ['code', 'severity', 'what', 'how'],
}
const REVIEW = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          reel_id: { type: 'string' },
          render_sha256: { type: ['string', 'null'] },
          decision: { type: 'string', enum: ['accept', 'fix_required'] },
          findings: { type: 'array', items: FINDING },
        },
        required: ['reel_id', 'render_sha256', 'decision', 'findings'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['results', 'notes'],
}
const STATUS = {
  type: 'object',
  properties: {
    exit_code: { type: 'integer' },          // last `reel job wait`: 0 settled, 2 worker failed/died, 124 still running
    id: { type: ['string', 'null'] },
    status: { type: 'string' },
    terminal: { type: 'boolean' },
    revision: { type: ['integer', 'null'] },
    next: { type: ['string', 'null'] },
    needs: { type: ['object', 'null'] },
    attempt: { type: ['object', 'null'] },
    error: { type: ['object', 'string', 'null'] },
    attention: { type: ['string', 'null'] },
    waiting: { type: ['array', 'null'] },
    counts: { type: ['object', 'null'] },
    findings_open: { type: ['integer', 'null'] },
    deliver: { type: ['object', 'null'] },
    log_tail: { type: ['string', 'null'] },
    start: { type: ['object', 'null'] },     // {exit_code, output} of `reel job start`
    done: { type: ['object', 'null'] },      // {exit_code, output} of the `reel job done` this turn ran
  },
  required: ['exit_code', 'status', 'terminal'],
}

// ------------------------------------------------------------------ runner turns

function runPrompt(rec) {
  const out = [`Reel job ${job}. Run these commands from the repository root exactly as written; do not change them.`]
  if (rec) {
    out.push(`0. Record the review: write the JSON below, verbatim, to ${rec.file} with the Write tool (if Write refuses ` +
      `because the file exists, Read it, then Write again). Then run once:\n   ${rec.done}\n` +
      `   reel-done-exit=0 means recorded; anything else means rejected: do not retry and change nothing, keep the exit ` +
      `code and output, and go on with step 1.\n\n${JSON.stringify(rec.envelope, null, 1)}\n`)
  }
  out.push(`1. ${START}`)
  out.push(`2. ${WAIT}\n   Give this Bash call a timeout of 600000 ms. While it prints reel-wait-exit=124, run it again ` +
    `(at most ${MAX_WAITS} times in a row; a Bash timeout counts as 124).`)
  out.push(`3. ${STATUS_CMD}`)
  out.push('Return the JSON printed by step 3, adding exit_code = the last reel-wait-exit number, ' +
    'start = {"exit_code": <reel-start-exit>, "output": <step 1 output, first 400 characters>}, ' +
    `done = ${rec ? '{"exit_code": <reel-done-exit>, "output": <its output>}' : 'null'}.`)
  return out.join('\n')
}

async function runTurn(i, rec) {
  let st = null
  try {
    st = await agent(runPrompt(rec), { agentType: 'reel-runner', schema: STATUS, phase: 'Run', label: `run #${i}` })
  } catch (e) {
    log(`runner turn ${i} failed: ${str(e && e.message, 200)}`)
    return null
  }
  if (!st || typeof st.status !== 'string' || typeof st.terminal !== 'boolean' || !Number.isInteger(st.exit_code)) return null
  if (st.id && st.id !== job) {
    log(`runner reported job ${str(st.id, 80)}, not ${job}: ignored`)
    return null
  }
  return st
}

function needOk(n) {
  return n && typeof n === 'object' && typeof n.step === 'string' && typeof n.need_id === 'string' &&
    ID_RE.test(n.need_id) && (n.revision == null || Number.isInteger(n.revision))
}

// ------------------------------------------------------------------ verify_style / plan

function noteFile(n, base) {
  const p = safePath(base) ? `${dirOf(base)}/${n.need_id}.note.txt` : null
  return p && safePath(p) ? p : `/tmp/reel-${job}-${n.need_id}.note.txt`
}

function needPrompt(n, base) {
  const note = noteFile(n, base)
  const done = `${CD}./reel job done ${job} ${n.step} --need ${n.need_id}${revFlag(n)} --note-file '${note}'; echo "reel-done-exit=$?"`
  const summary = Object.fromEntries(Object.entries(n).filter(([k]) => k !== 'assigned_ids' && k !== 'errors'))
  const errs = Array.isArray(n.errors) ? n.errors.filter(Boolean) : []
  return [
    `Reel job ${job} needs ${n.step} (need ${n.need_id}${Number.isInteger(n.revision) ? `, revision ${n.revision}` : ''}).`,
    `Print the full need (paths, request and notes, fixes, errors) with:\n   ${NEED_CMD}\nIf its output is cut off, redirect it to a file and Read that.`,
    `Summary: ${JSON.stringify(summary)}`,
    errs.length ? `Your previous answer to this need was rejected. Fix these first:\n- ${errs.join('\n- ')}` : '',
    `When your answer is in place, write a one-line note of what you did to ${note} with the Write tool (Read it first if it exists), ` +
      `then run exactly:\n   ${done}\nreel-done-exit=0 means accepted. Otherwise read the errors it printed, fix your answer and run it ` +
      'again (at most 3 times in all). Never put your own text into a shell command.',
    'Return a short summary of what you did and whether `reel job done` accepted it.',
  ].filter(Boolean).join('\n\n')
}

async function handOff(n, agentType, phase, label, base) {
  let r = null
  try {
    r = await agent(needPrompt(n, base), { agentType, phase, label })
  } catch (e) {
    r = null
  }
  log(r ? `${agentType}: ${str(r, 300)}` : `${agentType} returned nothing for ${n.need_id}`)
}

// ------------------------------------------------------------------ review

function reviewPrompt(n, ids) {
  const left = n.rounds_left_by_reel && typeof n.rounds_left_by_reel === 'object' ? n.rounds_left_by_reel : {}
  const errs = Array.isArray(n.errors) ? n.errors.filter(Boolean) : []
  return [
    `Review rendered reels of reel job ${job} for need ${n.need_id}` +
      `${Number.isInteger(n.plan_generation) ? ` (plan generation ${n.plan_generation})` : ''}.`,
    `Your reels: ${ids.join(', ')}. Return exactly one result for each of them and for no other reel.`,
    `Print your assignment (render_sha256, QC verdict, issues, file, sheet, qc paths, previous findings, pinned style and plan) with:\n   ${assignCmd(ids)}\n` +
      `If the printed need_id is not ${n.need_id}, the job has moved on: return {"results": [], "notes": "need changed"}.`,
    'Repair rounds left per reel: ' + ids.map((id) => `${id} ${Number.isInteger(left[id]) ? left[id] : '?'}`).join(', ') +
      '. A reel with 0 left will not be re-planned, but report every real defect anyway: every finding is recorded and escalated.',
    errs.length ? `The last answer to this need was rejected:\n- ${errs.join('\n- ')}` : '',
    'Copy render_sha256 exactly from the assignment (null when it has none). A reel whose qc_verdict is fail or error ' +
      'cannot be accepted. Return the review envelope as structured output.',
  ].filter(Boolean).join('\n\n')
}

async function review(n) {
  const all = (Array.isArray(n.assigned_ids) ? n.assigned_ids : []).filter((x) => typeof x === 'string')
  const ids = [...new Set(all.filter((x) => ID_RE.test(x)))]
  if (ids.length < all.length) {
    log(`${all.length - ids.length} assigned reel id(s) cannot be passed to a command safely; they stay unreviewed`)
  }
  const chunks = []
  for (let k = 0; k < ids.length; k += REVIEW_CHUNK) chunks.push(ids.slice(k, k + REVIEW_CHUNK))
  log(`reviewing ${ids.length} reel(s) in ${chunks.length} group(s) for ${n.need_id}`)
  const one = async (c, j, retry) => {
    try {
      const r = await agent(reviewPrompt(n, c), {
        agentType: 'qc-reviewer', schema: REVIEW, phase: 'Review',
        label: `review ${j + 1}/${chunks.length}${retry ? ' (retry)' : ''}`,
      })
      return r && Array.isArray(r.results) ? r : null
    } catch (e) {
      return null
    }
  }
  const got = await parallel(chunks.map((c, j) => async () => {
    const r = await one(c, j, false)
    if (r) return r
    log(`review group ${j + 1} (${c.join(', ')}) returned nothing: retrying once`)
    return one(c, j, true)
  }))

  // Merge: every finding is kept whatever the repair budget; Python decides what happens next.
  const byReel = new Map()
  const dropped = []
  const conflicts = []
  const notes = []
  chunks.forEach((c, j) => {
    const r = got[j]
    if (!r) return
    if (typeof r.notes === 'string' && r.notes.trim()) notes.push(r.notes.trim())
    const mine = new Set(c)
    for (const x of r.results) {
      if (!x || typeof x.reel_id !== 'string' || !mine.has(x.reel_id) ||
          (x.decision !== 'accept' && x.decision !== 'fix_required')) {
        dropped.push(str(x && x.reel_id, 64) || '?')
        continue
      }
      const findings = (Array.isArray(x.findings) ? x.findings : []).filter((f) => f && typeof f === 'object').map((f) => ({
        code: str(f.code, 80).trim() || 'UNSPECIFIED',
        severity: f.severity === 'minor' ? 'minor' : 'blocking',
        what: str(f.what, 2000),
        how: str(f.how, 2000),
      }))
      const sha = typeof x.render_sha256 === 'string' && x.render_sha256.trim() ? x.render_sha256.trim() : null
      const prev = byReel.get(x.reel_id)
      if (!prev) {
        byReel.set(x.reel_id, { reel_id: x.reel_id, render_sha256: sha, decision: x.decision, findings })
        continue
      }
      // one reviewer answered a reel twice: never an approval when either answer asks for a fix
      if (prev.decision !== x.decision) conflicts.push(x.reel_id)
      if (x.decision === 'fix_required') prev.decision = 'fix_required'
      prev.findings.push(...findings)
      if (!prev.render_sha256) prev.render_sha256 = sha
    }
  })
  const results = [...byReel.values()].map((r) => {
    const out = { reel_id: r.reel_id, decision: r.decision, findings: r.findings }
    if (r.render_sha256) out.render_sha256 = r.render_sha256
    return out
  })
  const missing = ids.filter((id) => !byReel.has(id))
  const nFix = results.filter((r) => r.decision === 'fix_required').length
  const nFind = results.reduce((k, r) => k + r.findings.length, 0)
  log(`review ${n.need_id}: ${results.length - nFix} accept, ${nFix} fix required, ${nFind} finding(s)` +
    (missing.length ? `; unreviewed (Python re-requests them): ${missing.join(', ').slice(0, 300)}` : ''))
  if (dropped.length) log(`ignored results for reels outside the reviewer's assignment: ${dropped.join(', ').slice(0, 300)}`)
  if (conflicts.length) log(`conflicting decisions for ${conflicts.join(', ')}: recorded as fix_required with all findings`)
  if (notes.length) log(`reviewer notes: ${notes.join(' | ').slice(0, 400)}`)

  const envelope = { schema_version: Number.isInteger(n.schema_version) ? n.schema_version : 1, job_id: job, need_id: n.need_id }
  if (Number.isInteger(n.revision)) envelope.expected_revision = n.revision
  if (Number.isInteger(n.plan_generation)) envelope.plan_generation = n.plan_generation
  envelope.results = results
  envelope.notes = notes.join(' | ').slice(0, 4000)
  const file = n.result_file
  // Python reads inbox/<need_id>.json by default; --result-file only when the path is single-quotable
  const fileFlag = safePath(file) ? ` --result-file '${file}'` : ''
  return {
    need_id: n.need_id,
    file,
    envelope,
    done: tailed(`./reel job done ${job} review --need ${n.need_id}${revFlag(n)}${fileFlag}`, 'reel-done-exit', 30),
  }
}

// ------------------------------------------------------------------ main loop

const blocked = (error, last) => {
  log(`stopped: ${error}`)
  return { status: 'blocked', terminal: false, error, job, last: last || null }
}

let last = null
let record = null
let failures = 0
const handed = {}
for (let i = 1; i <= MAX_STEPS; i++) {
  const st = await runTurn(i, record)
  if (!st) {
    failures++
    if (failures >= MAX_FAILURES) return blocked(`the reel-runner failed ${failures} turns in a row; rerun reel-job with the same job id to resume`, last)
    continue                                  // a pending review record is kept: recording it again is idempotent
  }
  const sent = record
  record = null
  last = st
  if (sent && !st.done && !st.terminal && st.needs && st.needs.need_id === sent.need_id) {
    failures++                                // the runner skipped step 0: send the same result again
    if (failures >= MAX_FAILURES) return blocked(`the review result for ${sent.need_id} was never recorded; rerun reel-job with the same job id`, st)
    log(`the review result for ${sent.need_id} was not recorded: sending it again`)
    record = sent
    continue
  }
  if (sent && st.done && st.done.exit_code !== 0) {
    log(`the review result was not recorded (reel-done-exit ${st.done.exit_code}): ${str(st.done.output, 300)}; continuing from the job's current state`)
  }
  if (st.terminal) {
    const d = st.deliver || {}
    log(`${st.status}: ${(d.files || []).length} reel(s) delivered` + (d.shortfall ? `, ${d.shortfall} short of the request` : '') +
      (d.unresolved ? `, ${d.unresolved} unresolved finding(s)` : '') +
      (st.status === 'failed' && st.error ? ` (${str(typeof st.error === 'string' ? st.error : st.error.cause, 300)})` : ''))
    return { job, ...st }
  }
  if (st.exit_code === 124) {
    failures = 0
    log(`worker still busy (${st.status}${st.attempt && st.attempt.stage ? `, ${st.attempt.stage}` : ''}): waiting on`)
    continue
  }
  const n = st.needs
  if (!n) {
    if (st.status === 'waiting:sample' || st.status === 'waiting:source') {
      log(`${st.status}: add the missing input with \`reel job add ${job} --${st.status.slice(8)} FILE\`, then rerun reel-job`)
      return { job, ...st }
    }
    // the worker died or failed without a terminal state (or a need was withdrawn by `reel job add`):
    // the next turn starts a new attempt, which resumes from the last valid step
    failures++
    const why = st.error ? str(typeof st.error === 'string' ? st.error : st.error.cause, 300)
      : st.exit_code !== 0 ? `reel job wait exited ${st.exit_code}` : `no need open at ${st.status}`
    log(`worker stopped without settling the job (${st.status}): ${why}`)
    if (failures >= MAX_FAILURES) {
      return blocked(`the worker stopped ${failures} times in a row without progress (${why}); see \`reel job status ${job}\` ` +
        'and the attempt log, then rerun reel-job with the same job id', st)
    }
    continue
  }
  if (!needOk(n)) {
    failures++
    log(`malformed need in the runner's report: ${str(JSON.stringify(n), 200)}`)
    if (failures >= MAX_FAILURES) return blocked('the runner kept reporting a malformed need; see `reel job status`', st)
    continue
  }
  failures = 0
  if (i === MAX_STEPS) break                  // no turn left to act on an answer: hand back resumable
  handed[n.need_id] = (handed[n.need_id] || 0) + 1
  if (handed[n.need_id] > MAX_SAME_NEED) {
    return blocked(`need ${n.need_id} (${n.step}) is still open after ${MAX_SAME_NEED} hand-offs; see needs.errors in ` +
      `\`reel job status ${job}\`, then rerun reel-job to try again`, st)
  }
  if (Array.isArray(n.errors) && n.errors.length) log(`${n.need_id}: the last answer was rejected: ${str(n.errors[0], 300)}`)
  if (n.step === 'verify_style') {
    log(`style needs verification (${n.need_id})`)
    await handOff(n, 'sample-analyst', 'Verify style', 'verify style', n.style)
  } else if (n.step === 'plan') {
    log(n.only || n.repair_round ? `re-plan for repair round ${n.repair_round || '?'} on ` +
      `${Array.isArray(n.only) && n.only.length ? n.only.join(', ') : 'all reels'}` : 'planning reels')
    await handOff(n, 'reel-planner', 'Plan', n.only || n.repair_round ? 'apply fixes' : 'plan reels', n.plan)
  } else if (n.step === 'review') {
    if (typeof n.result_file !== 'string' || !n.result_file) return blocked(`review need ${n.need_id} carries no result_file`, st)
    record = await review(n)
  } else {
    return blocked(`unknown need step ${str(n.step, 40)} (${n.need_id})`, st)
  }
}
return blocked('workflow step budget exhausted; rerun reel-job with the same job id to resume', last)
