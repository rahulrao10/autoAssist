"""decision_log.py -- records (and prints) every decision together with its reason.

Each entry is a small dict:  stage, key, decision, reason, details.
The log is (1) printed live in the terminal, (2) saved as JSON, and
(3) re-used by codegen.py to write the comments inside the generated script.
"""
import time


class DecisionLog:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.entries = []
        self._stage = "setup"
        self._t0 = time.perf_counter()

    def say(self, text=""):
        """Print a plain line (only when verbose)."""
        if self.verbose:
            print(text, flush=True)

    def section(self, number, total, title):
        """Print a banner like  [3/7] DATA DIAGNOSIS + CLEANING  and remember the stage."""
        self._stage = title
        self.say("")
        self.say("=" * 70)
        self.say("[%d/%d] %s" % (number, total, title))
        self.say("=" * 70)

    def add(self, key, decision, reason="", **details):
        """Store one decision and print it. `key` is a short id used by codegen."""
        self.entries.append({"stage": self._stage, "key": key, "decision": decision,
                             "reason": reason, "details": details})
        line = "  - " + decision
        if reason:
            line += "   (why: %s)" % reason
        self.say(line)

    def find(self, key):
        """Return the first entry with this key (or None)."""
        for e in self.entries:
            if e["key"] == key:
                return e
        return None

    def reason(self, key, default=""):
        e = self.find(key)
        return e["reason"] if e else default
