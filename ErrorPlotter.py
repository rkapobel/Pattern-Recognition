#!/usr/bin/python
import matplotlib.pyplot as plot

def plotErrorsByDegree(degrees, errors):
    plot.plot(degrees, errors[0], "b-o", label="Training")
    plot.plot(degrees, errors[1], "r-o", label="Test")
    plot.legend(loc='upper right', shadow=True)
    plot.axes().set_title("$Degree$ $influence$")
    plot.axes().set_xlabel("$M$")
    plot.axes().set_ylabel("$E_{drms}$")
    plot.yscale("log", nonposy='clip')
    plot.show()

def plotErrorsByLogLambda(lambdas, errors):
    plot.plot(lambdas, errors[0], "b-o", label="Training")
    plot.plot(lambdas, errors[1], "r-o", label="Test")
    plot.legend(loc='upper right', shadow=True)
    plot.axes().set_title("$\lambda$ $influence$")
    plot.axes().set_xlabel("$log(\lambda)$")
    plot.axes().set_ylabel("$E_{rms}$")
    plot.yscale("log", nonposy='clip')
    plot.show()

def plotOriginalVsEstimated(fOriginal, fEstimated, data, f1Name, degree, reg):
    plot.plot(data, [fOriginal(x) for x in data], "b-", label=f1Name)
    plot.plot(data, [fEstimated(x) for x in data], "r-", label="Estimated")
    plot.legend(loc='upper right', shadow=True)
    plot.axes().set_title("$" + f1Name + "$ $vs$ $Polynom$ $degree$: $" + str(degree) + "$ $\lambda$: $" + str(reg) + "$")
    plot.axes().set_xlabel("$x$")
    plot.axes().set_ylabel("$y$")
    plot.ylim(-1.1, 1.1)
    plot.show()