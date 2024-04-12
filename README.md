The accompanying software can reproduce all figures in the associated paper,
"Guarantees of confidentiality via Hammersley-Chapman-Robbins bounds." This
repository also provides LaTeX and BibTeX sources for replicating the paper.

The main files in the repository are the following:

``tex/paper.pdf``
PDF version of the paper

``tex/paper.tex``
LaTeX source for the paper

``tex/paper.bib``
BibTeX source for the paper

``tex/fairmeta.cls``
LaTeX document class file

``tex/logometa.pdf``
PDF image of Meta's logo

``codes/testers.py``
Python script for reproducing the figures after having run ``codes/trainer.py``

``codes/trainer.py``
Python script for training small neural nets on data sets, CIFAR-10 and MNIST

``codes/cifar10_model.py``
Python module defining a small neural net for the data set, CIFAR-10

``codes/mnist_model.py``
Python module defining a small neural net for the data set, MNIST

Be sure to change the directories, ``/datasets01/``, to wherever the data sets
are being stored. Both ``codes/trainer.py`` and ``codes/testers.py`` default to
``/datasets01/`` for all data sets.

To reproduce all figures, run the following commands:

``cd codes``

``python trainer.py MNIST``

``python trainer.py CIFAR10``

``python testers.py MNIST``

``python testers.py MNIST --limit``

``python testers.py CIFAR10``

``python testers.py CIFAR10 --limit``

``python testers.py Swin_T``

``python testers.py Swin_T --limit``

``python testers.py ResNet18``

``python testers.py ResNet18 --limit``

********************************************************************************

License

This hcrbounds software is licensed under the LICENSE file (the MIT license) in
the root directory of this source tree.
