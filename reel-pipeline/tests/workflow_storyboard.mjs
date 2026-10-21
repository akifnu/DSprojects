// Execute the saved workflow with a stubbed runner; no real agents or media work.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../../.claude/workflows/reel-job.js', import.meta.url), 'utf8')
  .replace('export const meta =', 'const meta =');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const calls = [];
const need = { step: 'approve_storyboard', agent: 'user', need_id: 'approve_storyboard-3', revision: 7,
  storyboard_sha256: 'a'.repeat(64), page: '/tmp/storyboard/index.html', sheet: '/tmp/storyboard/contact-sheet.jpg' };
const agent = async (_prompt, opts) => {
  calls.push(opts.agentType);
  assert.equal(opts.agentType, 'reel-runner', 'human approval was delegated to an agent');
  return { id: 'fixture', status: 'needs:approve_storyboard', terminal: false, exit_code: 0, needs: need };
};
const result = await new AsyncFunction('args', 'agent', 'log', source)({ job: 'fixture' }, agent, () => {});
assert.deepEqual(calls, ['reel-runner'], 'workflow retried instead of returning to the user');
assert.equal(result.requires_user_approval, true);
assert.equal(result.terminal, false);
assert.deepEqual(result.needs, need);
console.log('Workflow returned the exact storyboard need for user approval.');
