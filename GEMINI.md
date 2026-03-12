# GEMINI.md - Persistent Mandates

## Commit Rule
- **Commit every change**: Always commit code changes immediately after applying them. 
- Provide a clear, concise commit message explaining the "why" and "what".
- After each commit, run `git status` to verify.

## Technical Mandates
- **Gripper Control**: Use `GripperController` class for all gripper operations.
- **Task Manager Stability**: Always use `wait_drl_ready()` or a 1.0s delay when switching between DRL tasks to avoid Alarm 2007.
- **Modbus Polling**: Always poll Register 284 (Moving) for physical gripper movement confirmation.
