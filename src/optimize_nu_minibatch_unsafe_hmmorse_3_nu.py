###################################
# @name : AD_Poisson_ratio_implicit_diff_neighbor.py
# @author : mzu
# @created date : 22/02/22
# @function : generate value and gradient of poisson ratio 
# @ref: https://colab.research.google.com/github/google/jax-md/blob/main/notebooks/implicit_differentiation.ipynb#scrollTo=9YB8Qr2nOL5B
###################################
import time
import sys
import numpy as onp
import json
import os
import argparse
from scipy.optimize import fsolve
import gc

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, vmap, value_and_grad, grad
from jax.example_libraries import optimizers

from jax_md import space, energy, util, quantity, elasticity, partition, smap

from jaxopt.implicit_diff import custom_root

from minimization import run_minimization_while_nl_unsafe_fn

from utils import diameters_to_sigma_matrix, vector2dsymmat

f32 = jnp.float32
f64 = jnp.float64

Array = util.Array
maybe_downcast = util.maybe_downcast
DisplacementOrMetricFn = space.DisplacementOrMetricFn

def box_at_packing_fraction(sigmas, Ns, phi, dimension):
  sphere_volume_2d = lambda s, n: (jnp.pi / f32(4)) * n * s**2
  sphere_volume_3d = lambda s, n: (jnp.pi / f32(6)) * n * s**3
  if dimension == 2:
    sphere_volume = sphere_volume_2d
  elif dimension == 3:
    sphere_volume = sphere_volume_3d
  
  sphere_volume_total = jnp.sum(vmap(sphere_volume, in_axes=(0,0))(jnp.array(sigmas), jnp.array(Ns)))
  return (sphere_volume_total / phi) ** (1/dimension)

@jit
def check_c(C):
  C_dict = elasticity.extract_elements(C)
  cxxxx = C_dict["cxxxx"]
  cyyyy = C_dict["cyyyy"]
  cxxyy = C_dict["cxxyy"]
  cxyxy = C_dict["cxyxy"]
  cyyxy = C_dict["cyyxy"]
  cxxxy = C_dict["cxxxy"]
  C_matrix = jnp.array([[cxxxx, cxxyy, 2.0*cxxxy],[cxxyy, cyyyy, 2.0*cyyxy], [2.0*cxxxy, 2.0*cyyxy, 4.0*cxyxy]])
  eigens = jnp.linalg.eigvalsh(C_matrix)
  positive_C = jnp.all(jnp.linalg.eigvals(C_matrix)>0)
  return positive_C

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = onp.mean([d[key] for d in dict_list], axis=0)
    return mean_dict

def dict_sum(dict_list):
    sum_dict = {}
    for key in dict_list[0].keys():
        sum_dict[key] = sum(d[key] for d in dict_list)
    return sum_dict

def write_file(filename, val):
    i, loss, nus, num_rigids, params = val
    p_D = params["diameters_seed"]
    p_B = params["B_seed"]
    with open(filename, 'a') as f:
        print('{:6d}\t{:.16f}\t{:.16f}\t{:6d}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}'.format(
            i, loss, nus, num_rigids, p_D[0], p_D[1], p_B[0], p_B[1], p_B[2]), file=f)

def set_cutoff(params, alpha):
    Ds = params["diameters_seed"]
    def func_to_solve(x, Etol=1e-6, **kwargs):
        return jnp.exp(-2. * alpha * x) - 2. * jnp.exp(-alpha * x) + Etol

    func = jit(func_to_solve)
    root = fsolve(func, 1.0)
    return root

@jit
def harmonic_morse(dr: Array,
        epsilon: Array=5.0,
        alpha: Array=5.0,
        sigma: Array=1.0,
        k: Array=50.0, **kwargs) -> Array:
  U = jnp.where(dr < sigma,
               0.5 * k * (dr - sigma)**2 - epsilon,
               epsilon * (jnp.exp(-2. * alpha * (dr - sigma)) - 2. * jnp.exp(-alpha * (dr - sigma)))
               )
  return jnp.array(U, dtype=dr.dtype)

def setup(params,
          N=128,
          dimension=2,
          nspecies=2,
          alpha=6.0,
          ronset=2.0,
          rcutoff=2.5,
          k=50.0,
          phi=1.0,
          box_size=None,
          dt_start=0.001,
          dt_max=0.04):
  """ Set up the system and return a function that calculates P((D,B), R_init)
      Differentiating over this function will use implicit diff for the minimization.
  """
  
  N_s = int(N // nspecies)
  Ns = N_s * jnp.ones(nspecies, dtype=int)
  if box_size == None:
    box_size = box_at_packing_fraction(params["diameters_seed"], Ns, phi, dimension)
  displacement, shift = space.periodic(box_size)

  species_seed = jnp.arange(nspecies)
  species_vec = jnp.repeat(species_seed, N_s)

  rcut = set_cutoff(params, alpha)
  rcut_max = f32(rcut + jnp.max(params["diameters_seed"]))

  def energy_function(params):
    sigma_matrix = diameters_to_sigma_matrix(params["diameters_seed"])
    B_matrix = vector2dsymmat(params["B_seed"])
    return harmonic_morse_cutoff_pair(displacement,
                             species=species_vec, alpha=alpha, k=k,
                             sigma=sigma_matrix, epsilon=B_matrix)

  neighformat = partition.OrderedSparse
  def energy_hm_fn_nl(params):
    sigma_matrix = diameters_to_sigma_matrix(params["diameters_seed"])
    B_matrix = vector2dsymmat(params["B_seed"])
    ronset = f32(rcut_max*0.4 + jnp.max(sigma_matrix))
    rcutoff = f32(rcut_max + jnp.max(sigma_matrix))
    return smap.pair_neighbor_list(
                           energy.multiplicative_isotropic_cutoff(harmonic_morse, r_onset=ronset, r_cutoff=rcutoff),
                           space.canonicalize_displacement_or_metric(displacement),
                           species=species_vec, alpha=alpha, k=k,
                           sigma=sigma_matrix, epsilon=B_matrix)

  neighbor_list_fn = partition.neighbor_list(displacement, box_size,
                           r_cutoff=rcut_max, dr_threshold=0.3, capacity_multiplier=1.25)
  R_tmp = random.uniform(random.PRNGKey(0), (N, dimension), minval=0.0, maxval=box_size, dtype=f64)
  nbrs = neighbor_list_fn.allocate(R_tmp)

  def solver_nl_unsafe(R, params):
   new_energy_fn_nl = energy_hm_fn_nl(params)
   R_final, maxgrad, nbrs_final, niter = run_minimization_while_nl_unsafe_fn(
                                                             new_energy_fn_nl,
                                                             nbrs,
                                                             R, shift,
                                                             min_style=2,
                                                             dt_start=dt_start,
                                                             dt_max=dt_max)
   return R_final, (nbrs_final, maxgrad)

  def optimality_fn_nl_unsafe(R, p):
    o_energy_fn_nl = energy_hm_fn_nl(p)
    o_nbrs = nbrs.update(R)
    return quantity.force(o_energy_fn_nl)(R, neighbor=o_nbrs)
  decorated_solver_nl_unsafe = custom_root(optimality_fn_nl_unsafe, has_aux=True)(solver_nl_unsafe)

  def run_nl_unsafe_imp(params_dict, R_init):
   R_final, minimize_info = decorated_solver_nl_unsafe(R_init, params_dict)
   f_energy_fn_nl = energy_hm_fn_nl(params_dict)
   f_nbrs = nbrs.update(R_final)
   emt_fn = elasticity.athermal_moduli(f_energy_fn_nl, check_convergence=True)
   C, converg = emt_fn(R_final, box_size, neighbor=f_nbrs)
   measurement = elasticity.extract_isotropic_moduli(C)['nu']
   positive_c = check_c(C)
   return measurement, (positive_c, converg, minimize_info)

  return run_nl_unsafe_imp, box_size

def optimize_fn1(run_fn,
        param_init,
        simkey,
        resfile,
        logfile,
        start_learning_rate=1e-3,
        target_measurement=-0.1,
        nsamples=10,
        opt_steps=100,
        step_start=0):

    opt_int, opt_update, get_params = optimizers.rmsprop(start_learning_rate)

    def cond_fn(check):
      positive_c, converg, minimize_info = check
      nbrs_final = minimize_info[0]
      fmax = minimize_info[1]
      overflow = nbrs_final.did_buffer_overflow
      safe_nbrs = jnp.logical_not(overflow)
      minimize_success = jnp.logical_and(fmax<1.e-12, safe_nbrs)
      stable_c = jnp.logical_and(positive_c, converg)
      return jnp.logical_and(minimize_success, stable_c)

    def loss_fn(params, Rs):
     sumloss = 0.0
     rigids = []
     R_finals = []
     nbrs_finals = []
     measurements = []
     for i in range(nsamples):
        measurement, check = run_fn(params, Rs[i])
        rigid = cond_fn(check)
        loss2 = lax.cond(rigid, (), lambda _: (measurement-target_measurement)**2, (), lambda _: 0.0)
        sumloss += loss2
        rigids += [rigid]
        measurement = jnp.where(rigid, measurement, 0.0)
        measurements += [measurement]
     num_rigids = sum(rigids)
     mean_loss = jnp.where(num_rigids>0, sumloss/num_rigids, 0.0)
     sum_measurement = sum(measurements)
     mean_measurement = jnp.where(num_rigids>0, sum_measurement/num_rigids, 0.0)
     return mean_loss, (num_rigids, mean_measurement)

    vg_fn = value_and_grad(loss_fn, has_aux=True)

    def update_step(i, Rs, opt_state):
        params = get_params(opt_state)
        (loss, aux), g = vg_fn(params, Rs)
        num_rigid, mean_nus = aux
        new_state = lax.cond(num_rigid>0, (), lambda _: opt_update(i, g, opt_state), (), lambda _: opt_state)
        gc.collect()
        return new_state, (loss, num_rigid, mean_nus, g)        

    opt_state = opt_int(param_init)
    params = get_params(opt_state)
    for k in range(opt_steps):
      simkey, split = random.split(simkey)
      splits = random.split(split, num=nsamples)
      Rinits = vec_gen_Rinit(splits)
      opt_state, auxes = jit(update_step)(k, Rinits, opt_state)
      loss_tmp = auxes[0]
      rigids_tmp = auxes[1]
      nus_tmp = auxes[2]
      grads = auxes[3]
      steps = step_start + k + 1
      params = get_params(opt_state)
      write_file(resfile, (steps, loss_tmp, nus_tmp, rigids_tmp, params))
      write_file(logfile, (steps, loss_tmp, nus_tmp, rigids_tmp, grads))
    params = get_params(opt_state)
    return params

def optimize_fn2(run_fn,
        param_init,
        simkey,
        resfile,
        logfile,
        start_learning_rate=1e-3,
        target_measurement=-0.1,
        nsamples=2,
        opt_steps=2,
        step_start=0):

    opt_int, opt_update, get_params = optimizers.rmsprop(start_learning_rate)

    def cond_fn(check):
      positive_c, converg, minimize_info = check
      nbrs_final = minimize_info[0]
      fmax = minimize_info[1]
      overflow = nbrs_final.did_buffer_overflow
      safe_nbrs = jnp.logical_not(overflow)
      minimize_success = jnp.logical_and(fmax<1.e-12, safe_nbrs)
      stable_c = jnp.logical_and(positive_c, converg)
      return jnp.logical_and(minimize_success, stable_c)

    def loss_fn(params, R):
      measurement, check = run_fn(params, R)
      rigid = cond_fn(check)
      return jnp.where(rigid, (measurement-target_measurement)**2, 0.0), rigid

    vg_loss = value_and_grad(loss_fn, has_aux=True)

    def update_step(i, R_inits, opt_state):
        params = get_params(opt_state)
        zero_g = {"diameters_seed":jnp.zeros_like(params["diameters_seed"]), "B_seed":jnp.zeros_like(params["B_seed"])}

        rigids = []
        losses = []
        gs = []
        for j in range(nsamples):
          vals, grad = vg_loss(params, R_inits[j])
          loss2, rigid = vals
          g = lax.cond(rigid, (), lambda _: grad, (), lambda _: zero_g)
          losses += [loss2]
          rigids += [rigid]
          gs += [g]

        def true_fn(losses, gs):
            losses = jnp.array(losses)
            mean_losses = sum(losses) / num_rigid
            mean_g = dict_sum(gs)
            mean_g["diameters_seed"] /= num_rigid
            mean_g["B_seed"] /= num_rigid
            return opt_update(i, mean_g, opt_state), mean_g, mean_losses

        def false_fn():
            return opt_state, zero_g, 0.0

        num_rigid = sum(rigids)
        new_state, mean_g, mean_losses = lax.cond(num_rigid>0, (), lambda _: true_fn(losses, gs), (), lambda _: false_fn())
        return new_state, (mean_losses, num_rigid, mean_g)

    opt_state = opt_int(param_init)

    for k in range(opt_steps):
      simkey, split = random.split(simkey)
      splits = random.split(split, num=nsamples)
      Rinits = vec_gen_Rinit(splits)
      opt_state, auxes = jit(update_step)(k, Rinits, opt_state)
      loss_tmp = auxes[0]
      rigids_tmp = auxes[1]
      grads = auxes[2]
      steps = k + step_start + 1
      params = get_params(opt_state)
      write_file(resfile, (steps, loss_tmp, rigids_tmp, params))
      write_file(logfile, (steps, loss_tmp, rigids_tmp, grads))
    params = get_params(opt_state)
    return params

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keyseed', action='store', type=int, help='key seed')
parser.add_argument('-nu', '--targetnu', action='store', type=float, help='target poisson ratio')
parser.add_argument('-f', '--optfunc', action='store', type=int, help='optimization function')
parser.add_argument('-n', '--nsamples', action='store', type=int, help='number of samples')
parser.add_argument('-l', '--learnrate', action='store', type=float, help='learning rate')
parser.add_argument('-s', '--optsteps', action='store', type=int, help='optimization steps')

args = parser.parse_args()

key_seed = args.keyseed
nu_target = args.targetnu
opt_fn = args.optfunc
nsamples = args.nsamples
start_learning_rate = args.learnrate
opt_steps = args.optsteps

key = random.PRNGKey(key_seed)

N = 128
dimension = 2
n_species=2

resfile="/home/sw23/jax/src/data/loss_hm_mean"+str(opt_fn)+"_ns"+str(nsamples)+"_lr"+str(start_learning_rate)+"_nu"+str(nu_target)+"_3.dat"
if os.path.isfile(resfile):
    with open(resfile, 'r') as f:
        for line in f:
            pass
        last_line = line
        sp = last_line.split("\t")
        step_start = sp[0]
        diameters_seed = jnp.array([sp[3], sp[4]], dtype=f32)
        B_seed = jnp.array([sp[5], sp[6], sp[7]], dtype=f32)
else:
    step_start = 0
    diameters_seed = jnp.array([1.0, 1.0])
    B_seed = jnp.array([0.01, 0.01, 0.01])
param_dict = {"diameters_seed":diameters_seed, "B_seed":B_seed}

boxfile="/home/sw23/jax/src/data/loss_hm_mean"+str(opt_fn)+"_ns"+str(nsamples)+"_lr"+str(start_learning_rate)+"_nu"+str(nu_target)+"_3.box"
if os.path.isfile(boxfile):
    with open(boxfile, 'r') as f:
        box_size = f.readline()
        box_size = f32(box_size)
else:
    box_size = None

run, box_size = setup(param_dict, N, dimension, box_size=box_size)

logfile="/home/sw23/jax/src/data/loss_hm_mean"+str(opt_fn)+"_ns"+str(nsamples)+"_lr"+str(start_learning_rate)+"_nu"+str(nu_target)+"_3.log"

if opt_fn==1:
    optimize_func = optimize_fn1
else:
    optimize_func = optimize_fn2

gen_Rinit = jit(lambda x: random.uniform(x, (N, dimension), minval=0.0, maxval=box_size, dtype=f64))
vec_gen_Rinit = vmap(gen_Rinit)

params = optimize_func(run, param_dict, key, resfile, logfile, 
        start_learning_rate=start_learning_rate, target_measurement=nu_target,
        nsamples=nsamples, opt_steps=opt_steps, step_start=int(step_start))

if os.path.isfile(boxfile):
    pass
else:
    with open(boxfile, 'w') as f:
        print('{:.16f}'.format(box_size), file=f)

