# RAMP starting kit on classification of variable stars from light curves

[![Build Status](https://travis-ci.org/ramp-kits/variable_stars.svg?branch=master)](https://travis-ci.org/ramp-kits/variable_stars)

_Authors: Balázs Kégl, Marc Moniez, Alex Gramfort, Djalel Benbouzid, Mehdi Cherti_

Most stars emit light steadily in time, but a small fraction of them has a variable light curve: light emission versus time. We call them variable stars. The light curves are usually periodic and highly regular. There are essentially two reasons why light emission can vary. First, the star itself can be oscillating, so its light emission varies in time. Second, the star that seems a single point at Earth (because of our large distance) is actually a binary system: two stars that orbit around their common center of gravity. When the orbital plane is parallel to our line of view, the stars eclipse each other periodically, creating a light curve with a charateristic signature. Identifying, classifying, and analyzing variable stars are hugely important for calibrating distances, and making these analyses automatic will be crucial in the upcoming sky survey projects such as LSST.

The challenge in this RAMP is to design an algorithm to automatically classify variable stars from light curves.

#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](variable_stars_starting_kit.ipynb).

#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.
