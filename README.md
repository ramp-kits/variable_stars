# RAMP starting kit on classification of variable stars from light curves

[![Build Status](https://travis-ci.org/ramp-kits/variable_stars.svg?branch=master)](https://travis-ci.org/ramp-kits/variable_stars)

Most stars emit light steadily in time, but a small fraction of them has a
variable light curve: light emission versus time. We call them variable stars.
The light curves are usually periodic and highly regular. There are essentially
two reasons why light emission can vary. First, the star itself can be
oscillating, so its light emission varies in time. Second, the star that seems
a single point at Earth (because of our large distance) is actually a binary
system: two stars that orbit around their common center of gravity. When the
orbital plane is parallel to our line of view, the stars eclipse each other
periodically, creating a light curve with a characteristic signature.
Identifying, classifying, and analyzing variable stars are hugely important for
calibrating distances, and making these analyses automatic will be crucial in
the upcoming sky survey projects such as LSST.

The challenge in this RAMP is to design an algorithm to automatically classify variable stars from light curves.

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with the
[dedicated notebook](variable_stars_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
