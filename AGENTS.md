# Agent Notes

## Codex sandbox compatibility on this host

This host may fail every normal sandboxed command with:

    error building bubblewrap command: mountinfo path is not absolute

Root cause: Snap creates nsfs entries in `/proc/self/mountinfo` whose mount-root field
looks like `mnt:[4026532918]`. Codex 0.155's new mount-isolation parser expects that
field to be an absolute path and aborts before Bubblewrap starts.

The persistent compatibility fix is already enabled in the user Codex configuration:

    codex features enable use_legacy_landlock

After changing that setting, restart the Codex VS Code extension/app-server before
testing because the running server does not reload sandbox settings. Verify with:

    codex features list | rg use_legacy_landlock

Then run a normal, non-escalated read command. If the same error remains after restart,
update the Codex extension/CLI and re-test. Do not unmount `/run/snapd/ns/*.mnt`;
those are system-wide Snap namespace mounts and are not project state.

For a session that was already running when the feature was changed, use narrowly scoped
approved elevated commands until restart. Do not use `danger-full-access` as a
persistent workaround.
