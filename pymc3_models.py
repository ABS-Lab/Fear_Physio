from typing import Any, Union

import pymc3 as pm
from sklearn import preprocessing

def MeanModel(
        df,
        outcome
):
    """
       models fear score as a normal distribution

        @param df: pandas data frame with colums of outcome and iv_string
        @param outcome: string of out (y) column
        @return: pymc3 model
   """

    df = df.dropna()
    fear = df[outcome].values

    with pm.Model() as fear_model:
        # priors for each physio metric
        mu = pm.Normal('mu', mu=0., sd=1)
        sd = pm.HalfCauchy('eps', 5.)
        b = pm.Normal('b', mu=0., sd=sd, observed=fear)

    return fear_model

def RegressOnePredictor(
        df,
        outcome,
        iv_string
):
    """
    simple regression relating iv to dv returns mcmc chain and pymc3 model

    @param df: pandas data frame with colums of outcome and iv_string
    @param outcome: string of out (y) column
    @param iv_string: string for x column
    @param n_samples: number of samples to do in mcmc chains
    @return: pymc3 model, mcmc trace
    """
    # clean data
    df = df.dropna()

    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values

    with pm.Model() as flat_model:
        beta_iv = pm.Normal('beta_' + iv_string, mu=0, tau=.1)
        # priors for error
        b = pm.Normal('mu_b', mu=0., sd=100)

        fear_est = beta_iv * iv + b

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return flat_model


def RegressOnePredictor_Situation(
        df,
        outcome,
        iv_string
):
    """
       regression: betas per different conditiosn -> mcmc chain model

        @param df: pandas data frame with colums of outcome and iv_string
        @param outcome: string of out (y) column
        @param iv_string: string for x column
        @param n_samples: number of samples to do in mcmc chains
        @return: pymc3 model, mcmc trace
   """
    df = df.dropna()
    video_condition = df.video_condition.values.astype(int) - 1  # trial by trial indexes of condition
    n_condition = len(df.video_condition.unique())  # n different conditions
    print(n_condition)

    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values

    with pm.Model() as fear_model:
        # priors for each physio metric
        iv_beta = pm.Normal('beta_' + iv_string, mu=0, sd=1, shape=n_condition)
        # priors for error
        b = pm.Normal('b', mu=0., sd=1, shape=n_condition)

        fear_est = iv_beta[video_condition] * iv + b[video_condition]  #

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return fear_model

def RegressOnePredictor_Idiographic(
        df,
        outcome,
        iv_string
):
    """
       regression: betas per different participant -> mcmc chain model

        @param df: pandas data frame with colums of outcome and iv_string
        @param outcome: string of out (y) column
        @param iv_string: string for x column
        @return: pymc3 model, mcmc trace
   """

    df = df.dropna()
    trials_participants = df['participant_num']
    num_participants = trials_participants.nunique()
    par_idx = preprocessing.LabelEncoder().fit_transform(trials_participants)  # n different conditions

    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values

    with pm.Model() as fear_model:
        # priors for each physio metric
        iv_beta = pm.Normal('beta_' + iv_string, mu=0, sd=1, shape=num_participants)
        # priors for error
        b = pm.Normal('b', mu=0., sd=1, shape=num_participants)

        fear_est = iv_beta[par_idx] * iv + b[par_idx]  #

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return fear_model


def RegressOnePredictor_SubjectHierarchical(
        df,
        outcome,
        iv_string
):
    """
        hierarchical regression with subject and group level estimates -> pymc3 model,trace

        @param df: pandas data frame with colums of outcome and iv_string
        @param outcome: string of out (y) column
        @param iv_string: string for x column
        @param n_samples: number of samples to do in mcmc chains
        @return: pymc3 model, mcmc trace
    """
    # clean data
    df = df.dropna()

    # get participant info
    trials_participants = df['participant_num']
    num_participants = trials_participants.nunique()
    par_idx = preprocessing.LabelEncoder().fit_transform(trials_participants)
    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values
    with pm.Model() as fear_model_method:
        # priors for each physio metric
        iv_mu = pm.Normal('mu_' + iv_string, mu=0, sd=1)
        iv_sd = pm.HalfCauchy('sd_' + iv_string, 5)

        # priors for error
        mu_b = pm.Normal('mu_b', mu=0., sd=1)
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=100)

        beta_iv_offset = pm.Normal('beta_offset_' + iv_string, mu=0, sd=1, shape=num_participants)
        beta_iv = pm.Deterministic('beta_' + iv_string, iv_mu + beta_iv_offset * iv_sd)

        b_offset = pm.Normal('b_offset', mu=0, sd=1, shape=num_participants)
        b = pm.Deterministic('b', mu_b + b_offset * sigma_b)

        fear_est = beta_iv[par_idx] * iv + b[par_idx]  #

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return fear_model_method


def RegressOnePredictor_SituationSubjectHierarchical(
        df,
        outcome,
        iv_string
):
    """
       runs condition specific hierarchical regression for subject and group level estimates
       @param df: pandas data frame with colums of outcome and iv_string
       @param outcome: string of out (y) column
       @param iv_string: string for x column
       @param n_samples: number of samples to do in mcmc chains
       @return: pymc3 model, mcmc trace
    """
    df = df.dropna()
    video_condition = df.video_condition.values.astype(int) - 1  # trial by trial indexes of condition
    n_condition = len(df.video_condition.unique())  # n different conditions

    # get participant info
    trials_participants = df['participant_num']
    num_participants = trials_participants.nunique()
    par_idx = preprocessing.LabelEncoder().fit_transform(trials_participants)

    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values

    with pm.Model() as fear_model:
        # priors for each physio metric
        iv_mu = pm.Normal('mu_' + iv_string, mu=0, sd=1, shape=n_condition)
        iv_sd = pm.HalfCauchy('sd_' + iv_string, 5, shape=n_condition)

        # priors for error
        mu_b = pm.Normal('mu_b', mu=0., sd=1, shape=n_condition)
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=100, shape=n_condition)

        beta_iv_offset = pm.Normal('beta_offset_' + iv_string, mu=0, sd=1, shape=(num_participants, n_condition))
        beta_iv = pm.Deterministic('beta_' + iv_string, iv_mu + beta_iv_offset * iv_sd)

        b_offset = pm.Normal('b_offset', mu=0, sd=1, shape=(num_participants, n_condition))
        b = pm.Deterministic('b', mu_b + b_offset * sigma_b)

        fear_est = beta_iv[par_idx, video_condition] * iv + b[par_idx, video_condition]  #

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return fear_model


def RegressOnePredictor_Idiographic(
        df,
        outcome,
        iv_string
):
    """
       regression: betas per different conditiosn -> mcmc chain model

        @param df: pandas data frame with colums of outcome and iv_string
        @param outcome: string of out (y) column
        @param iv_string: string for x column
        @param n_samples: number of samples to do in mcmc chains
        @return: pymc3 model, mcmc trace
   """

    df = df.dropna()
    trials_participants = df['participant_num']
    num_participants = trials_participants.nunique()
    par_idx = preprocessing.LabelEncoder().fit_transform(trials_participants)  # n different conditions

    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values

    with pm.Model() as fear_model:
        # priors for each physio metric
        iv_beta = pm.Normal('beta_' + iv_string, mu=0, sd=1, shape=num_participants)
        # priors for error
        b = pm.Normal('b', mu=0., sd=1, shape=num_participants)

        fear_est = iv_beta[par_idx] * iv + b[par_idx]  #

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return fear_model


def RegressOnePredictor_SituationSubjectIndependent(
        df,
        outcome,
        iv_string
):
    """
       models regressions by per participant AND situation independently
       @param df: pandas data frame with colums of outcome and iv_string
       @param outcome: string of out (y) column
       @param iv_string: string for x column
       @param n_samples: number of samples to do in mcmc chains
       @return: pymc3 model, mcmc trace
    """
    df = df.dropna()
    video_condition = df.video_condition.values.astype(int) - 1  # trial by trial indexes of condition
    n_condition = len(df.video_condition.unique())  # n different conditions

    # get participant info
    trials_participants = df['participant_num']
    num_participants = trials_participants.nunique()
    par_idx = preprocessing.LabelEncoder().fit_transform(trials_participants)

    # get values of variables of interest
    iv = df[iv_string].values
    fear = df[outcome].values

    with pm.Model() as fear_model:
        # priors for each physio metric
        iv_beta = pm.Normal('beta_' + iv_string, mu=0, sd=1, shape=(num_participants, n_condition))
        # priors for error
        b = pm.Normal('mu_b', mu=0., sd=1, shape=(num_participants, n_condition))

        fear_est = iv_beta[par_idx, video_condition] * iv + b[par_idx, video_condition]  #

        eps = pm.HalfCauchy('eps', 5.)
        nu = pm.Exponential('nu', 1 / 29)
        fear_like = pm.StudentT('obs_est', mu=fear_est, nu=nu, sd=eps, observed=fear)

    return fear_model

def SampleModel(
        pm_model,
        n_samples,
        cores=2,
        target_accept=.9,
        tune=2000
):
    """
           samples given model
           @param pm_model: instantiated pymc3 model
           @param n_samples: length of mcmc chain
           @param cores: how many cores to run the mcmc sampler on default 2
           @param target_accept: acceptance probability default .9
           @param tune: number of tuning samples default = 2000
           @return: mcmc trace
        """
    with pm_model:
        trace = pm.sample(n_samples, cores=cores, target_accept=target_accept, tune=tune)
        return trace
