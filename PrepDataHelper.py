def CalcAkaikeWeights(dIC):
    rel_likelihood = np.exp(-.5*dIC)
    normalization_factor = np.sum(rel_likelihood)
    return rel_likelihood/normalization_factor

def GetPhobiaScoreList(df,phobia_string, survey_data):
    score_arr = []
    for index, row in df.iterrows():
        phobia_score = survey_data.loc[survey_data.Q11 == row.participant_num][phobia_string].values
        if(len(phobia_score)>0):
            score_arr.append(phobia_score[0])
        else:
            score_arr.append(np.NAN)
    return score_arr


def TakeAwayGroupsLessThenX(df, min_groups=4):
    participants_with_more_then_two_trials = []
    for idx, grouped in df.groupby('participant_num'):
        if (len(grouped) > min_groups):
            participants_with_more_then_two_trials.append(idx)
    return df[df['participant_num'].isin(participants_with_more_then_two_trials)]

