# COE notes

## tmpdir
The `pymagnitude` converter uses `tempfile.gettempdir()` which defaults to `/tmp`. The COE preferse `/scratch`.

Set that via `TMPDIR=/scratch` environment variable.

## script usage
The main script is `train_and_get_neighbors`.

Before using, modify the script to point `$code` at the script directory. You may also change `$vecname` if you want a different basename for the output files.

Usage:
```
scale/train_and_get_neighbors \
  ../ko/ko-en.vector.extended \
  ../ko/ko-en.vector.en.scaled \
  ../ko/ko-dict-mturk.txt
```
where the args are: foreign vectors, English vectors, set of word pairs.
This will generate (by default)
* `vecs.src.magnitude`
* `vecs.tgt.magnitude`
* `testset` with tab-separated word pairs
