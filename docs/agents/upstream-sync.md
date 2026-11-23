# Upstream sync

Upstream is `CrissFerrada/Poly-X-Microplastics`. This tree began as a fork of it
and has since been translated (Spanish identifiers → English), restructured
(`polyx/` → `src/polyx/`), and stripped of the consumer-distribution track. That
makes it a fork git cannot merge with: a 3-way merge needs common text, and after
translation there is none.

**Measured, not assumed** — a trial `git merge upstream/main` on `origin/main`
(2026-11-20) produced **29 conflicts**: 18 deleted-by-us (the old `polyx/` tree
upstream still edits), 5 added-by-them landing on renamed paths, 5 both-modified
in translated files, 1 add/add on `LICENSE`. Asking git to rename-detect harder
(`-X find-renames=20%`) gave 27 conflicts but started 3-way-merging Spanish
identifier hunks into English files, which is worse. Divergence of the touched
counterparts: `report.py` 82 %, `trainer/state.py` 75 %, `trainer/runner.py`
71 %, `trainer/pages/model.py` 56 %, and `report_html.py` is now a 7-line shim
over a 12-module `src/polyx/gui/report/` package, so upstream's changes to it
have nowhere to land.

So upstream is a **donor, not a branch to merge**. Porting is done by reading
`git show upstream/main:<path>` and rewriting the behaviour in this tree's idiom.

## Closing the ledger

When the ports for one upstream range are done, record the review with an
ancestry-only merge:

```
git merge -s ours upstream/main -m "reviewed upstream through <sha>: took X, deliberately not taking Y"
```

`-s ours` writes the merge commit and takes **no content** (tree hash before and
after is identical — verified), after which `git merge upstream/main` reports
"Already up to date" until upstream commits again. It is not a lie about
content: it is a record that this range was read, and the message is where the
decision about each commit goes. Without it, the same 29 conflicts come back
every time someone tries to sync.

## Reviewed range: through `aaf25c0` (2026-11-20)

Seven commits beyond the last sync point (`6e749d1`).

| Commit | What it is | Status here |
|---|---|---|
| `2082796` | Numbered, exportable tables and figures (`exportables.py`) | **Ported** — `src/polyx/gui/report/exportables.py` |
| `b692222` | YOLOv8-P2, RT-DETR and RF-DETR engines beside YOLOv8/YOLO11 | Not taken — see below |
| `f5e6feb` | Install optional components from the UI, RF-DETR first | Not taken — depends on the engine layer |
| `b51de62` | MIT licence | Taken by hand: `LICENSE` carries Ferrada's notice verbatim plus one line for the maintainer |
| `abd9211`, `04f2cf7` | MIT declared in both READMEs | Not taken: this tree's README is rewritten; the licence statement is already there |
| `aaf25c0` | Copyright notice corrected to 2025–2026 | Taken by hand, same file |

### Why the engine commits are not taken yet

`b692222` and `f5e6feb` add a fifth and sixth trainer to the comparison while
M007 (SAM 2.1 teacher → YOLO-seg student) is mid-execution, and adding engines
is the one change that would reorder that milestone. They also depend on
`polyx/core/motores.py` and `motor_rfdetr.py`, which have no counterpart here
(`src/polyx/core/detect/` is a different abstraction: a protocol with `yolo.py`
and `sam.py` behind it). Porting them is a design job, not a move.

They stay on the list because RF-DETR is Apache-2.0: it is a concrete way out of
the AGPL position that holding ultralytics as the only engine puts this
distribution in (M007/S05).

## Rules for the next sync

- Port per feature, one commit each, with the upstream SHA in the message.
- Rename and translate as part of the port; do not leave Spanish identifiers or
  Spanish msgids behind.
- Update `CHANGELOG.md`, this file, and the catalogs (`tools/i18n/cli.py
  extract && update && compile`) in the same commit — the gate's translation
  audit fails otherwise.
- Check behavioural differences deliberately: upstream's `exportables` numbered
  the per-particle measure tables, which this tree does not.
