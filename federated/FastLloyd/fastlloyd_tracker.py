from baselines.paper_common import PaperTracker


class FastLloydTracker(PaperTracker):
    def __init__(self):
        super().__init__("FastLloyd (bounded, Gaussian MSA)")


ConvergenceTracker = FastLloydTracker
