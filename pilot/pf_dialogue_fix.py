"""Rerun the dialogue no-hint cells with the neutral first turn.

FormGen.inputs renders self.turn directly, and set_form leaves it empty for
the no-hint cells, so the Gemma dialogue baseline had an empty first user
turn.  This patches inputs to substitute NEUTRAL_TURN when the turn is empty
and runs only the two dialogue no-hint cells into a patch file.
"""

import sys

import pilot.prompt_forms_gen as g
from pilot.prompt_forms import NEUTRAL_TURN

_orig = g.FormGen.inputs


def inputs(self, images, text):
    if not self.turn:
        self.turn = NEUTRAL_TURN
    return _orig(self, images, text)


g.FormGen.inputs = inputs
g.prompts = lambda r: [("dialogue", "absent", "none", r["absent_category"], None),
                       ("dialogue", "sufficient", "none", None, None)]
out = sys.argv[1] if len(sys.argv) > 1 else "runs/pf_gemma_dialogue_fix.jsonl"
sys.argv = ["x", "--out", out]
g.main()
