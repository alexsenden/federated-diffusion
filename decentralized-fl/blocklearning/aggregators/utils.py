from flwr.server.strategy.fedavg import aggregate
from flwr.server.strategy.fedmedian import aggregate_median


def weighted_fed_avg(submissions, model_size, weights_loader, avg_weights):
    # total_weights = np.sum(avg_weights)
    # new_weights = np.zeros(model_size)

    # for i, submission in enumerate(submissions):
    #     (_, _, _, weights_cid) = submission
    #     weights = weights_loader.load(weights_cid)
    #     new_weights += weights * (avg_weights[i] / total_weights)

    weights_results = [
        (weights_loader.load(weights_cid), 100)
        for _, _, _, weights_cid in submissions
    ]
    return aggregate_median(weights_results)
