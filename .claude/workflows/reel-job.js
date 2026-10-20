export const meta = {
  name: 'reel-job',
  description: 'Uploaded sample + long-form footage → scrutinised style, planned, rendered and QC-reviewed reels',
  whenToUse: 'After `reel job new` has registered uploads; args: {job: "<job id>"}',
  phases: [
    { title: 'Advance', detail: 'mechanical steps by the production line (ingest, analysis, transcription, editing copies, batch render + QC)' },
    { title: 'Verify style', detail: 'sample-analyst checks the measured style against the real sample' },
    { title: 'Plan', detail: 'reel-planner picks the moments and writes plan.json' },
    { title: 'Review', detail: 'qc-reviewers check rendered reels in parallel and decide fixes' },
  ],
}

const JOB = args && args.job
if (!JOB) throw new Error('args.job (from `reel job new`) is required')
const MAX_STEPS = 16           // advance ↔ agent hand-offs before giving up
const REVIEW_CHUNK = 8         // reels per qc-reviewer

const STATUS = {
  type: 'object',
  properties: {
    status: { type: 'string' },
    busy: { type: 'boolean' },
    needs: { type: ['object', 'null'] },
    deliver: { type: ['object', 'null'] },
    rounds: { type: 'number' },
    error: { type: 'string' },
  },
  required: ['status'],
}
const REVIEW = {
  type: 'object',
  properties: {
    fixes: { type: 'array', items: { type: 'object', properties: { reel: { type: 'string' }, what: { type: 'string' }, how: { type: 'string' } }, required: ['reel', 'what', 'how'] } },
    ok: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['fixes', 'ok'],
}

const advance = (i) => agent(
  `Advance reel job ${JOB}: run \`./reel job advance ${JOB}\` detached and poll \`./reel job status ${JOB}\` until busy is false, as your instructions describe. Return the final status JSON fields.`,
  { agentType: 'reel-runner', schema: STATUS, phase: 'Advance', label: `advance #${i}` })

let status = null
for (let i = 1; i <= MAX_STEPS; i++) {
  status = await advance(i)
  if (!status) throw new Error('runner died')
  if (status.error) { log(`error: ${status.error.slice(0, 300)}`); return status }
  if (status.status === 'done' || status.status.startsWith('blocked')) break
  const n = status.needs
  if (!n) continue
  if (n.step === 'verify_style') {
    log('style needs verification')
    await agent(`Reel job ${JOB} needs verify_style. Details: ${JSON.stringify(n)}`,
      { agentType: 'sample-analyst', phase: 'Verify style', label: 'verify style' })
  } else if (n.step === 'plan') {
    log(n.fixes ? `re-plan for fixes on ${n.only || 'all'}` : 'planning reels')
    await agent(`Reel job ${JOB} needs plan. Details: ${JSON.stringify(n)}`,
      { agentType: 'reel-planner', phase: 'Plan', label: n.fixes ? 'apply fixes' : 'plan reels' })
  } else if (n.step === 'review') {
    const reels = n.reels || []
    const chunks = []
    for (let k = 0; k < reels.length; k += REVIEW_CHUNK) chunks.push(reels.slice(k, k + REVIEW_CHUNK))
    log(`reviewing ${reels.length} reel(s) in ${chunks.length} group(s); ${n.rounds_left} fix round(s) left`)
    const verdicts = (await parallel(chunks.map((c, j) => () => agent(
      `Review these reels of job ${JOB} (plan + manifest: ${n.manifest}; the style is data/assets/<plan.style>/style.json). ` +
      `${n.rounds_left > 0 ? '' : 'No fix rounds left: only report fixes for clear defects; they will be escalated to the user.'}\n` +
      JSON.stringify(c), { agentType: 'qc-reviewer', schema: REVIEW, phase: 'Review', label: `review ${j + 1}/${chunks.length}` })))).filter(Boolean)
    const fixes = n.rounds_left > 0 ? verdicts.flatMap(v => v.fixes) : []
    const notes = verdicts.map(v => v.notes).filter(Boolean).join(' | ')
    if (notes) log(`reviewer notes: ${notes.slice(0, 400)}`)
    await agent(
      `Record the review for reel job ${JOB}: run \`./reel job done ${JOB} review --note ${JSON.stringify((notes || `${fixes.length} fixes`).slice(0, 200))} --fixes '${JSON.stringify(fixes).replace(/'/g, "'\\''")}'\` and return the status JSON.`,
      { agentType: 'reel-runner', schema: STATUS, phase: 'Review', label: 'record review' })
  }
}
return status
