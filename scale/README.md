# COE notes

## tmpdir
The `pymagnitude` converter uses `tempfile.gettempdir()` which defaults to `/tmp`. The COE preferse `/scratch`.

Set that via `TMPDIR=/scratch` environment variable.
