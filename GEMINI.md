# GEMINI.md - Persistent Mandates

## Commit Rule
- **Commit every change**: Always commit code changes immediately after applying them. 
- Provide a clear, concise commit message explaining the "why" and "what".
- After each commit, run `git status` to verify.

## Session Summary (March 12, 2026)
- **Resolved Cascading Alarms**: Fixed Alarm 2007 (Task Error), 2008 (Conflict), and 2016 (Type Error) by aligning DRL with Python 3 `bytes` and correcting serial port closing logic.
- **Architectural Refactor**: Switched `startup.py` to `MultiThreadedExecutor` to prevent service deadlocks.
- **Deterministic Orchestration**: Modified `start_order_stack.py` to run `startup.py` as a synchronous blocking step before spawning application nodes.
- **Hard Reset Logic**: Implemented a Manual -> Autonomous mode toggle in the startup sequence to clear latched robot controller alarms automatically.
- **Internalized Polling**: Moved gripper "Moving" status polling inside the DRL interpreter to reduce ROS 2 overhead.

## Next Task
- **Verification of Pick & Action Nodes**: Test the end-to-end ordering flow. Verify that the `pick.py` node correctly identifies items via YOLO and that `action.py` executes the pouring sequence without further DRL collisions.
