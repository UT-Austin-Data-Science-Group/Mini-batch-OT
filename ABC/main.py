import argparse
import os
import tempfile
import pickle
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import pyabc
from pyabc.sge import nr_cores_available
import utils
from scipy.stats import invgamma


np.random.seed(1)
random.seed(1)


def distance_fn(type, k=2, m=32):

    if type == "bombOT":
        return lambda x, y: utils.BoMbOT(x["data"], y["data"], k=k, m=m)
    elif type == "mOT":
        return lambda x, y: utils.mOT(x["data"], y["data"], k=k, m=m)
    else:
        raise ValueError("Distance type should be bombOT or  mOT")


def save_results(history, dirname):
    # Create directory that will contain the results
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for it in range(history.max_t + 1):
        # Save the posterior distribution at each ABC iteration
        filename = "posterior_it=" + str(it) + ".csv"
        df, w = history.get_distribution(m=0, t=it)
        df["weight"] = w
        df.to_csv(os.path.join(dirname, filename))

        # Save extended information at each iteration, including weighted distances that the parameter samples achieve
        filename = "info_it=" + str(it) + ".csv"
        df = history.get_population_extended(m=0, t=it)
        df.to_csv(os.path.join(dirname, filename))

    # Save information on the evolution of epsilon, the number of sample attempts per iteration and the iteration times
    filename = "all_populations.csv"
    df = history.get_all_populations()
    # df['times'] = np.insert(times, 0, 0)
    df.to_csv(os.path.join(dirname, filename))


def plot_posterior(param, dim, n_obs, n_it, n_particles, types, labels, k, m):
    # Matplotlib settings
    plt.rcParams["lines.linewidth"] = 1

    directory = os.path.join(
        "results",
        param
        + "_dim="
        + str(dim)
        + "_n_obs="
        + str(n_obs)
        + "_n_particles="
        + str(n_particles)
        + "_n_it="
        + str(n_it)
        + "_k="
        + str(k)
        + "_m="
        + str(m),
    )

    # Plot true posterior pdf
    fig = plt.figure(0, figsize=(4, 2))
    with open(os.path.join(directory, "true_posterior"), "rb") as f:
        post_samples = pickle.load(f)
    pyabc.visualization.plot_kde_1d(
        pd.DataFrame({"post_samples": post_samples}),
        np.ones(post_samples.shape[0]) / post_samples.shape[0],
        xmin=0,
        xmax=10,
        ax=plt.gca(),
        x="post_samples",
        color="darkgray",
        linestyle="--",
        numx=1000,
        label="True posterior",
    )
    t = np.linspace(0, 10, 1000)
    plt.fill_between(t, plt.gca().lines[0].get_ydata(), facecolor="gray", alpha=0.4)

    # Plot ABC posteriors
    for i in range(len(types)):
        df = pd.read_csv(os.path.join(directory, types[i], "all_populations.csv"))
        max_it = df["t"].iloc[-1]
        df = pd.read_csv(os.path.join(directory, types[i], "posterior_it=" + str(max_it) + ".csv"))
        w = df["weight"].values
        w = w / np.sum(w)
        scale = df["scale"].values
        df = df[df.columns.difference(["weight"])]
        W2 = ot.emd2(ot.unif(post_samples.shape[0]), w, ot.dist(post_samples[:, None], scale[:, None]))
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=10,
            ax=plt.gca(),
            x="scale",
            numx=1000,
            label=labels[i] + " $W_2$=" + str(np.round(W2, 2)),
        )
        plt.fill_between(t, plt.gca().lines[-1].get_ydata(), facecolor=plt.gca().lines[-1].get_color(), alpha=0.2)
        plt.xlabel("$\sigma^2$", fontsize=12)
        plt.ylabel("density", fontsize=14)
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(directory, "abc_posteriors_k{}_m{}.pdf".format(k, m)), bbox_inches="tight")
    plt.close(fig)


def main(param, dim, n_obs, n_procs, n_it, n_particles, max_time, types, labels, k=2, m=32):
    np.random.seed(1)
    random.seed(1)
    # Create directory that will contain the results
    directory = os.path.join(
        "results",
        param
        + "_dim="
        + str(dim)
        + "_n_obs="
        + str(n_obs)
        + "_n_particles="
        + str(n_particles)
        + "_n_it="
        + str(n_it)
        + "_k="
        + str(k)
        + "_m="
        + str(m),
    )
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define data-generating parameters
    true_mean = np.random.normal(size=dim)
    true_scale = 4
    Sigma_likelihood = true_scale * np.eye(dim)
    # Define priors on the scale parameter
    alph = 1
    prior_args = {"scale": pyabc.RV("invgamma", alph)}
    prior = pyabc.Distribution(prior_args)

    # Generate observations
    observations = np.random.multivariate_normal(true_mean, Sigma_likelihood, size=n_obs)
    # Save the dataset of observations
    with open(os.path.join(directory, "dataset"), "wb") as f:
        pickle.dump(observations, f, pickle.HIGHEST_PROTOCOL)

    # Define parameters of the true posterior
    alph_post = alph + 0.5 * (n_obs * dim)
    beta_post = 1 + 0.5 * ((observations - true_mean) * (observations - true_mean)).sum()
    # Generate parameter samples from the true posterior
    post_samples = invgamma.rvs(a=alph_post, scale=beta_post, size=n_particles)
    # Save the result
    with open(os.path.join(directory, "true_posterior"), "wb") as f:
        pickle.dump(post_samples, f, pickle.HIGHEST_PROTOCOL)

    # Define generative model used in ABC to generate synthetic data
    def model(parameter):
        Sigma = (parameter["scale"]) * np.eye(dim)
        return {"data": np.random.multivariate_normal(true_mean, Sigma, size=n_obs)}

    times = []
    for i in range(len(types)):
        print("Running ABC-SMC with " + str(labels[i]) + " distance...")
        np.random.seed(1)
        random.seed(1)
        start = time.time()
        distance = distance_fn(types[i], k=k, m=m)
        abc = pyabc.ABCSMC(
            models=model,
            parameter_priors=prior,
            distance_function=distance,
            population_size=n_particles,  # nb of particles
            sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs),
            eps=pyabc.epsilon.QuantileEpsilon(alpha=0.5),
        )

        db_path = ("sqlite:///" +
                   os.path.join(tempfile.gettempdir(), "test.db"))
        abc_id = abc.new(db_path, {"data": observations})
        # Run ABC-SMC
        history = abc.run(minimum_epsilon=0.01, max_nr_populations=n_it, max_walltime=timedelta(seconds=max_time * 60.0))
        end = time.time()
        times.append(end - start)
        # Save results
        print("Done! Saving results for ABC-SMC with " + str(labels[i]) + " distance...")
        save_results(history, os.path.join(directory, types[i]))
    print(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2, help="data dimension")
    parser.add_argument("--n_obs", type=int, default=128, help="number of observations")
    parser.add_argument(
        "--n_procs",
        type=int,
        default=nr_cores_available(),
        help="number of processors to use for parallelization",
    )
    parser.add_argument("--n_it", type=int, default=10, help="number of ABC iterations")
    parser.add_argument("--n_particles", type=int, default=128, help="number of particles")
    parser.add_argument("--max_time", type=float, default=10.0, help="maximum running time (in min)")
    parser.add_argument("--k", type=int, default=2, help="the number of mini-batches")
    parser.add_argument("--m", type=int, default=16, help="the size of mini-batches")
    args = parser.parse_args()
    # Try different distances on ABC-SM
    k = args.k
    m = args.m
    test_types = ["mOT", "bombOT"]
    test_labels = ["m-OT", "BoMb-OT"]
    main(
        param="scale",
        dim=args.dim,
        n_obs=args.n_obs,
        n_procs=args.n_procs,
        n_it=args.n_it,
        n_particles=args.n_particles,
        max_time=args.max_time,
        types=test_types,
        labels=test_labels,
        k=k,
        m=m,
    )

    print("Plotting the final posterior distribution...")
    plot_posterior(
        param="scale",
        dim=args.dim,
        n_obs=args.n_obs,
        n_it=args.n_it,
        n_particles=args.n_particles,
        types=test_types,
        labels=test_labels,
        k=k,
        m=m,
    )
