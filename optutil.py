#!/usr/bin/python

#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Shai Revzen, U Penn, 2010

# ******NOTICE***************
# Based on: fmin from optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

"""
optutil includes several tweaks of Travis E. Oliphant's fmin from scipy's
optimize.py. In particular, it includes:

  fmin -- a drop in replacement of optimize.fmin that accepts an initial simplex
  
  fminIter -- a "iterator" version of fmin that generates the sample points
    from an iterator, freeing the user's code to be the main function rather
    than a callback.
    
  Additional functions:
  
  test_fminIter -- demonstrates the use of fminIter
"""
import numpy

__all__ = ['fmin','fminIter']

def wrap_function(function, args):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper

def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, simcallback=None):
    """Minimize a function using the downhill simplex algorithm.

    :Parameters:

      func : callable func(x,*args)
          The objective function to be minimized.
      x0 : ndarray of length D or (D+1) x D 
          Initial guess or initial simplex
      args : tuple
          Extra arguments passed to func, i.e. ``f(x,*args)``.
      callback : callable
          Called after each iteration, as callback(xk), where xk is the
          current parameter vector.
      simcallback : callable
          Called after each iteration, as simcallback(sim,fsim) with sim
          the (d+1,d) simplex and fsim the (d+1) function values

    :Returns: (xopt, {fopt, iter, funcalls, warnflag})

      xopt : ndarray
          Parameter that minimizes function.
      fopt : float
          Value of function at minimum: ``fopt = func(xopt)``.
      iter : int
          Number of iterations performed.
      funcalls : int
          Number of function calls made.
      warnflag : int
          1 : Maximum number of function evaluations made.
          2 : Maximum number of iterations reached.
      allvecs : list
          Solution at each iteration.

    *Other Parameters*:

      xtol : float
          Relative error in xopt acceptable for convergence.
      ftol : number
          Relative error in func(xopt) acceptable for convergence.
      maxiter : int
          Maximum number of iterations to perform.
      maxfun : number
          Maximum number of function evaluations to make.
      full_output : bool
          Set to True if fval and warnflag outputs are desired.
      disp : bool
          Set to True to print convergence messages.
      retall : bool
          Set to True to return list of solutions at each iteration.

    :Notes:

        Uses a Nelder-Mead simplex algorithm to find the minimum of
        function of one or more variables.

    """
    fcalls, func = wrap_function(func, args)
    x0 = numpy.asarray(x0,dtype=float)
    rank = len(x0.shape)
    if rank<2:
      N = len(x0)
    else:
      N = x0.shape[1]
    if rank==2 and x0.shape[0] != x0.shape[1]+1:
        raise ValueError, "Initial simplex must be of shape (d+1,d)."
    elif not -1 < rank <= 2:
        raise ValueError, "Initial guess must be a scalar, a sequence or a simplex."
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200
    if callback and not callable(callback):
        raise TypeError,"The callback parameter must be callable"
    if simcallback and not callable(simcallback):
        raise TypeError,"The simcallback parameter must be callable"
        
    rho = 1; chi = 2; psi = 0.5; sigma = 0.5;
    one2np1 = range(1,N+1)

    fsim = numpy.zeros((N+1,), float)
    if rank == 0:
        sim = numpy.zeros((N+1,), dtype=x0.dtype)
    elif rank == 2:
        sim = x0.copy()
        for k in xrange(fsim.size):
            fsim[k] = func(sim[k,:])
    else: 
        assert rank==1     
        sim = numpy.zeros((N+1,N), dtype=x0.dtype)        
        sim[0] = x0
        if retall:
            allvecs = [sim[0]]
        fsim[0] = func(x0)
        nonzdelt = 0.05
        zdelt = 0.00025
        for k in range(0,N):
            y = numpy.array(x0,copy=True)
            if y[k] != 0:
                y[k] = (1+nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k+1] = y
            f = func(y)
            fsim[k+1] = f

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim,ind,0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim,ind,0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (max(numpy.ravel(abs(sim[1:]-sim[0]))) <= xtol \
            and max(abs(fsim[0]-fsim[1:])) <= ftol):
            break

        xbar = numpy.add.reduce(sim[:-1],0) / N
        xr = (1+rho)*xbar - rho*sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1+rho*chi)*xbar - rho*chi*sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else: # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else: # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1+psi*rho)*xbar - psi*rho*sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink=1
                else:
                    # Perform an inside contraction
                    xcc = (1-psi)*xbar + psi*sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma*(sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim,ind,0)
        fsim = numpy.take(fsim,ind,0)
        if callback is not None:
            callback(sim[0])
        if simcallback is not None:
            simcallback(sim,fsim)
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of function evaluations has "\
                  "been exceeded."
    elif iterations >= maxiter:
        warnflag = 2
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % iterations
            print "         Function evaluations: %d" % fcalls[0]


    if full_output:
        retlist = x, fval, iterations, fcalls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist

def fminIter( x0, f0 = None, xtol=1e-4, ftol=1e-4):
  """Iterator implementing Nelder-Mead function optimization
  
    Runs a Nelder-Mead optimization in an iterator. This allows the user to 
    do anything she wants while "inside" the goal function. The optimization
    runs until tolerance limits are reached, but no iteration limit is enforced.
    To enforce a function call, just break out of the iterator loop... 
    
      x0 -- (D+1)xD -- Initial simplex
      f0 -- (D+1) -- Values at initial simplex, if known
      xtol -- termination tolerance for parameters
      ftol -- termination tolerance for values
      
    Example usage:
      def goalFunction(x):
        return (x[0]-0.5)**4 + x[1]**2 + (x[1]-1)*x[0]
      sim0 = array([[-1,-1],[-1,4],[4,-1]])
      opt = fminIter(sim0)
      val = opt.next()
      for x in opt:
        val[-1] = goalFunction( x )
      print "Minimum at ",val[0]," value ",val[1]
  """
  # Convert inputs into array form and verify shapes
  x0 = numpy.asarray(x0,dtype=float)
  if x0.ndim != 2:
    raise ValueError, "x0 must be rank 2; got rank %d" % x0.ndim
  if f0 is not None:
    f0 = numpy.asarray(f0,dtype=float)
    if f0.shape != x0.shape[:1]:
      raise IndexError, "f0 shape is %s which does not match x0 shape of %s" % (str(f0.shape),str(x0.shape))
  # Dimension        
  N = x0.shape[1]
  if x0.shape[0] != x0.shape[1]+1:
    raise ValueError, "Initial simplex must be of shape (d+1,d), got %s" % str(x0.shape)
    
  # Expose storage for function values
  val = [None]
  yield val

  # Algorithm parameters
  rho = 1; chi = 2; psi = 0.5; sigma = 0.5;

  # Intialize simplex
  sim = x0.copy()
  # If simplex values are available --> use them
  if f0 is not None:
    fsim = f0.copy()
    for k in numpy.isnan(fsim).nonzero()[0]:
      yield sim[k,:]
      fsim[k] = val[-1]
  else: # else get values at simplex
    fsim = numpy.zeros_like(sim[:,0])
    for k in xrange(fsim.size):
      yield sim[k,:]
      fsim[k] = val[-1]

  # sort simplex so sim[0,:] has the lowest function value
  ind = numpy.argsort(fsim)
  fsim = numpy.take(fsim,ind,0)
  sim = numpy.take(sim,ind,0)
  while ( max(numpy.ravel(abs(sim[1:]-sim[0]))) > xtol
      or  max(abs(fsim[0]-fsim[1:])) > ftol ):

    xbar = numpy.add.reduce(sim[:-1],0) / N
    xr = (1+rho)*xbar - rho*sim[-1]
    yield xr
    fxr = val[-1]
    doshrink = 0

    if fxr < fsim[0]:
      xe = (1+rho*chi)*xbar - rho*chi*sim[-1]
      yield xe
      fxe = val[-1]

      if fxe < fxr:
        sim[-1] = xe
        fsim[-1] = fxe
      else:
        sim[-1] = xr
        fsim[-1] = fxr
    else: # fsim[0] <= fxr
      if fxr < fsim[-2]:
        sim[-1] = xr
        fsim[-1] = fxr
      else: # fxr >= fsim[-2]
        # Perform contraction
        if fxr < fsim[-1]:
          xc = (1+psi*rho)*xbar - psi*rho*sim[-1]
          yield xc
          fxc = val[-1]

          if fxc <= fxr:
            sim[-1] = xc
            fsim[-1] = fxc
          else:
            doshrink=1
        else:
          # Perform an inside contraction
          xcc = (1-psi)*xbar + psi*sim[-1]
          yield xcc
          fxcc = val[-1]

          if fxcc < fsim[-1]:
            sim[-1] = xcc
            fsim[-1] = fxcc
          else:
            doshrink = 1

        if doshrink:
          for j in xrange(1,N+1):
            sim[j] = sim[0] + sigma*(sim[j] - sim[0])
            yield sim[j]
            fsim[j] = val[-1]

    # Sort the simplex
    ind = numpy.argsort(fsim)
    sim = numpy.take(sim,ind,0)
    fsim = numpy.take(fsim,ind,0)
  # END WHILE 
  # Store result in value mailbox
  val[:] = (sim[0],fsim[0])
  return

def test_fminIter():
  import numpy as np
  def goal(x):
    return (x[0]-0.5)**4 + x[1]**2 + (x[1]-1)*x[0]
  #sim0 = numpy.array([[-1,-1],[-1,4],[4,-1]])
  #sim0 = numpy.array([[-1,-1],[-1,-1],[-1,-1]]) + np.random.randn(3,2)
  sim0 = numpy.array([[-1,-1],[-1,-1],[-1,4]])
  opt = fminIter(sim0)
  val = opt.next()
  for x in opt:
      f = goal(x)
      print x,"-->",f
      val[-1]=f
  x,f = val
  print "Minimum at ",tuple(x),"with value",f

