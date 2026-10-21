"""menu.py -- step 6: print the ranked table and let the user pick a model."""


def format_table(results, metric_label):
    """Return the ranked model table as text."""
    head = "%-3s %-31s %-17s %-9s %-8s %-9s %-8s %s" % (
        "#", "Model", "CV (mean+-95%CI)", "Holdout", "Gap", "vs dummy", "Rating", "")
    lines = ["Metric: %s   (Gap = CV - holdout; vs dummy = CV gain over the baseline)" % metric_label,
             head, "-" * 92]
    for r in results:
        lines.append("%-3d %-31s %-17s %-9.4f %-+8.3f %-+9.3f %-8s" % (
            r.rank, r.spec.display, "%.4f+-%.4f" % (r.cv_mean, r.cv_ci), r.holdout, r.gap, r.gain, r.rating))
    lines.append("")
    for r in results:
        lines.append("  #%d %s: %s" % (r.rank, r.spec.display, r.rating_reason))
    return "\n".join(lines)


def format_menu(results):
    """Short ranked list shown right before the question (the full table was printed in step 5)."""
    return "\n".join("  %d) %-32s CV %.4f   [%s]" % (r.rank, r.spec.display, r.cv_mean, r.rating) for r in results)


def find_model(results, choice):
    """`choice` may be a rank number (1, '2'), a model key ('random_forest') or a display name."""
    text = str(choice).strip().lower()
    for r in results:
        if text == str(r.rank) or text == r.spec.key.lower() or text == r.spec.display.lower():
            return r
    return None


def choose_model(results, choice=None, input_fn=input, interactive=True, say=print):
    """Pick a model. choice given -> use it. Otherwise ask (Enter = top model)."""
    if choice is not None:
        r = find_model(results, choice)
        if r is None:
            raise ValueError("Unknown model '%s'. Options: %s" % (choice, [x.spec.key for x in results]))
        say("  Using model #%d: %s" % (r.rank, r.spec.display))
        return r
    if not interactive:
        say("  Non-interactive: using the top-ranked model, %s" % results[0].spec.display)
        return results[0]
    while True:
        try:
            answer = input_fn("Pick a model by number [1 = %s]: " % results[0].spec.display).strip()
        except EOFError:           # no keyboard attached -> fall back to the default
            answer = ""
        if answer == "":
            return results[0]
        r = find_model(results, answer)
        if r is not None:
            return r
        say("  Please type a number between 1 and %d." % len(results))
