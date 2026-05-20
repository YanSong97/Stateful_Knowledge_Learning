import numpy as np



def nq_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    if data_source in ["nq", "nq_search", "hotpotqa", "browse_camp_search", "gaia_search",
                       "webshaper_search", "webwalker_qa_search", "2WikiMultihopQA_rand1000", "Bamboogle", "frames", "GAIA",
                       "HotpotQA_rand1000", "Musique_rand1000", "NQ_rand1000", "PopQA_rand1000", "TriviaQA_rand1000", "xbench-deepsearch"]:
        import src.utils.reward_score.qa_em as qa_em
        res = qa_em.compute_score_em(solution_str, ground_truth, return_dict=True)

        # import pdb
        # pdb.set_trace()

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
