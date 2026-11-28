"""Ticket #122: one operating-point module, and every figure names its criterion.

There are two legitimate ways to score a detection run and they disagree by
construction: *localisation* asks only whether the particle was found, while
*with class* also requires the polymer. The module under test owns both, so a
reader cannot compute a third definition of "F1" by hand.

The run's own decision (``ImageResult.match``, ticket #120) is a third
quantity: it is the match the run performed at the threshold it used. The page
and the report read it through ``Score.from_decision``; only the hypothetical
threshold sweep re-matches, and it does so with the rule the run recorded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from polyx.core.detect.protocol import Instance  # noqa: E402
from polyx.core.detection_run import ImageResult, RunParams  # noqa: E402
from polyx.core.detection_run.operating_point import (  # noqa: E402
    CRITERION_LABELS,
    Criterion,
    Score,
    prf1,
    recommend,
    score,
)
from polyx.core.quality.metrics import match_image  # noqa: E402
from tests.match_fixtures import misclassified_result  # noqa: E402


def _det(
    class_id: int,
    conf: float,
    box: tuple[float, float, float, float],
    name: str | None = None,
) -> Instance:
    x1, y1, x2, y2 = box
    return Instance(
        class_id=class_id,
        class_name=name or ("PET" if class_id == 0 else "PP"),
        conf=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _confidence_fixture(tmp_path: Path) -> ImageResult:
    """Two true positives at different confidences and one false positive.

    At 0.90 one TP is kept and one particle is missed (F1 0.667). At 0.40 both
    are found and the stray box counts (F1 0.800). The best observed threshold
    is therefore the *lowest* one: a sweep that only tried the strongest
    detection would pick the wrong operating point.
    """
    return ImageResult(
        image_path=tmp_path / "plate.png",
        model_idx=0,
        predictions=[
            _det(0, 0.9, (10, 10, 50, 50)),
            _det(1, 0.8, (200, 200, 240, 240)),
            _det(0, 0.4, (100, 100, 140, 140)),
        ],
        gt=[
            _det(0, 1.0, (10, 10, 50, 50)),
            _det(0, 1.0, (100, 100, 140, 140)),
        ],
        has_gt=True,
    )


# ── The two criteria ─────────────────────────────────────────────────────


def test_localisation_credits_a_misclassified_match(tmp_path):
    """The box was on the particle: localisation found it (#224)."""
    result = misclassified_result(tmp_path / "plate.png")

    measured = score([result], Criterion.LOCALISATION, 0.25)

    # One correct pair and one well-placed wrong-class pair: both were found.
    assert (measured.tp, measured.fp, measured.fn, measured.miscls) == (2, 0, 0, 1)
    assert measured.precision == pytest.approx(1.0)
    assert measured.recall == pytest.approx(1.0)
    assert measured.f1 == pytest.approx(1.0)


def test_a_lone_misclassified_box_is_found_but_wrongly_labelled(tmp_path):
    """#224's reproduction: perfect geometry, wrong polymer, F1 was 0.0.

    Localisation measures whether the particle was found, so the pair is a
    true positive there and the score is perfect. With class it is a false
    positive of the predicted class and a false negative of the real one, and
    only that criterion may report zero.
    """
    image = tmp_path / "plate.png"
    prediction = _det(1, 0.9, (10, 10, 50, 50), name="PP")
    truth = _det(0, 1.0, (10, 10, 50, 50))
    match = match_image([prediction], [truth], iou_thr=0.5)
    result = ImageResult(
        image_path=image,
        model_idx=0,
        predictions=[prediction],
        gt=[truth],
        has_gt=True,
        tp=match.tp,
        fp=match.fp,
        fn=match.fn,
        miscls=match.miscls,
        match=match,
        match_iou_thr=0.5,
        match_use_mask_iou=False,
    )
    assert (match.tp, match.fp, match.fn, match.miscls) == (0, 0, 0, 1)

    localisation = Score.from_decision([result], Criterion.LOCALISATION, 0.25)
    with_class = Score.from_decision([result], Criterion.WITH_CLASS, 0.25)

    assert (localisation.tp, localisation.precision, localisation.recall) == (1, 1.0, 1.0)
    assert localisation.f1 == pytest.approx(1.0)
    assert (with_class.tp, with_class.fp, with_class.fn) == (0, 1, 1)
    assert with_class.f1 == pytest.approx(0.0)


def test_with_class_counts_it(tmp_path):
    result = misclassified_result(tmp_path / "plate.png")

    measured = score([result], Criterion.WITH_CLASS, 0.25)

    assert (measured.tp, measured.fp, measured.fn, measured.miscls) == (1, 1, 1, 1)
    assert measured.precision == pytest.approx(0.5)
    assert measured.recall == pytest.approx(0.5)
    assert measured.f1 == pytest.approx(0.5)


def test_the_same_run_counters_give_both_criteria_without_rematching(tmp_path):
    """A finished run carries its decision; the criterion only picks its place."""
    result = misclassified_result(tmp_path / "plate.png")

    localisation = Score.from_decision([result], Criterion.LOCALISATION, 0.25)
    with_class = Score.from_decision([result], Criterion.WITH_CLASS, 0.25)

    # Same decision, two questions: the located-but-wrong-class pair is a
    # true positive for one criterion and an error for the other.
    assert (localisation.tp, localisation.fp, localisation.fn) == (2, 0, 0)
    assert (with_class.tp, with_class.fp, with_class.fn) == (1, 1, 1)
    assert localisation.f1 == pytest.approx(1.0)
    assert with_class.f1 == pytest.approx(0.5)
    # Both describe the same decision: no second matching pass invented it.
    assert localisation.miscls == with_class.miscls == 1


def test_prf1_is_the_one_arithmetic_helper():
    assert prf1(2, 1, 1) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert prf1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_from_counts_folds_the_same_pair_for_both_criteria():
    """``miscls`` is a subset of localisation TP and is inside with-class FP/FN."""
    localisation = Score.from_counts(Criterion.LOCALISATION, 0.5, tp=3, fp=2, fn=1, miscls=4)
    with_class = Score.from_counts(Criterion.WITH_CLASS, 0.5, tp=3, fp=2, fn=1, miscls=4)

    assert (localisation.tp, localisation.fp, localisation.fn, localisation.miscls) == (
        7,
        2,
        1,
        4,
    )
    assert (with_class.tp, with_class.fp, with_class.fn, with_class.miscls) == (3, 6, 5, 4)
    assert localisation.tp - localisation.miscls == with_class.tp
    assert localisation.fp + localisation.miscls == with_class.fp
    assert localisation.fn + localisation.miscls == with_class.fn


# ── Recommending an operating point ──────────────────────────────────────


def test_recommend_names_its_criterion(tmp_path, selected_language):
    selected_language("en")
    result = misclassified_result(tmp_path / "plate.png")

    assert recommend([result]).criterion is Criterion.WITH_CLASS
    assert recommend([result], Criterion.LOCALISATION).criterion is Criterion.LOCALISATION
    assert recommend([result]).label == "with class"
    assert recommend([result], Criterion.LOCALISATION).label == "localisation"
    assert str(CRITERION_LABELS[Criterion.WITH_CLASS]) == "with class"


def test_recommend_returns_best_observed_threshold(tmp_path):
    result = _confidence_fixture(tmp_path)

    best = recommend([result], Criterion.WITH_CLASS)

    assert best.threshold == pytest.approx(0.4)
    assert best.f1 == pytest.approx(0.8)
    assert set(best.__dataclass_fields__) >= {
        "criterion",
        "threshold",
        "tp",
        "fp",
        "fn",
        "miscls",
        "precision",
        "recall",
        "f1",
    }


def test_recommend_sweeps_every_observed_threshold(tmp_path):
    """The best threshold is not skipped by a coarse candidate grid.

    Fifty-two distinct confidences exist, and the true maximum sits on a
    candidate the old ``[::len // 25]`` subsampling never visited (0.10): the
    false positive at 0.04 is cut at 0.10, while 0.11 already loses a true
    positive.
    """
    predictions = [_det(0, 0.04, (0, 0, 10, 10))]  # the false positive
    gt = []
    for index, confidence in enumerate(
        (round(0.10 + step * 0.01, 2) for step in range(51)), start=1
    ):
        x = float(index * 20)
        predictions.append(_det(0, confidence, (x, 0, x + 10, 10)))
        gt.append(_det(0, 1.0, (x, 0, x + 10, 10)))
    result = ImageResult(
        image_path=tmp_path / "plate.png",
        model_idx=0,
        predictions=predictions,
        gt=gt,
        has_gt=True,
    )
    assert len({round(p.conf, 2) for p in predictions}) == 52

    best = recommend([result], Criterion.WITH_CLASS)

    assert best.threshold == pytest.approx(0.10)
    assert best.f1 == pytest.approx(1.0)


def test_recommend_without_ground_truth_is_an_empty_score(tmp_path):
    result = _confidence_fixture(tmp_path)
    result.has_gt = False

    best = recommend([result])

    assert best.f1 == 0.0
    assert best.threshold == 0.0


def test_run_params_stay_the_run_s_threshold(tmp_path):
    """The page passes the run's confidence; nothing here reads it by accident."""
    result = _confidence_fixture(tmp_path)
    measured = score([result], Criterion.LOCALISATION, RunParams().conf)
    assert measured.threshold == RunParams().conf
    assert measured.fp == 1  # the stray box survives at 0.25


# ── One owner for the arithmetic ─────────────────────────────────────────


def test_the_old_prf1_copies_are_gone():
    """No reader keeps its own division of the counters."""
    pytest.importorskip("PySide6")

    import inspect

    from polyx.core import report_metrics, stats
    from polyx.gui.detector.pages import compare
    from polyx.gui.report import sections

    assert not hasattr(report_metrics, "_pr_f1")
    for module in (compare, sections, stats):
        source = inspect.getsource(module)
        assert "_pr_f1" not in source, module.__name__
        # The sweeps re-match through the operating-point module, never locally.
        assert "match_image(" not in source, module.__name__
