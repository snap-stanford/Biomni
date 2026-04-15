import time

import numpy as np

from reinvent_scoring.scoring.score_summary import FinalSummary
from reinvent_chemistry.logging import fraction_valid_smiles


class ConsoleMessage:

    def create(self, start_time, n_steps, step, score_summary: FinalSummary,
               agent_likelihood, prior_likelihood, augmented_likelihood):
        mean_score = np.mean(score_summary.total_score)
        time_message = self._time_progress(start_time, n_steps, step, score_summary.scored_smiles, mean_score)
        score_message = self._score_profile(score_summary.scored_smiles, agent_likelihood, prior_likelihood,
                                            augmented_likelihood, score_summary.total_score)
        score_breakdown = self._score_summary_breakdown(score_summary)
        message = time_message + score_message + score_breakdown
        return message

    def _time_progress(self, start_time, n_steps, step, smiles, mean_score):
        time_elapsed = int(time.time() - start_time)
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        valid_fraction = fraction_valid_smiles(smiles)
        message = (f"\n Step {step}   Fraction valid SMILES: {valid_fraction:4.1f}   Score: {mean_score:.4f}   "
                   f"Sample size: {len(smiles)}   "
                   f"Time elapsed: {time_elapsed}   "
                   f"Time left: {time_left:.1f}\n")
        return message

    def _score_profile(self, smiles, agent_likelihood, prior_likelihood, augmented_likelihood, score):
        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()
        message = "     ".join(["  Agent", "Prior", "Target", "Score"] + ["SMILES\n"])
        for i in range(min(10, len(smiles))):
            message += f'{agent_likelihood[i]:6.2f}    {prior_likelihood[i]:6.2f}    ' \
                       f'{augmented_likelihood[i]:6.2f}    {score[i]:6.2f} '
            message += f"     {smiles[i]}\n"
        return message

    def _score_summary_breakdown(self, score_summary: FinalSummary):
        message = "   ".join([c.name for c in score_summary.profile])
        message += "\n"
        for i in range(min(10, len(score_summary.scored_smiles))):
            for summary in score_summary.profile:
                message += f"{summary.score[i]}   "
            message += "\n"
        return message
