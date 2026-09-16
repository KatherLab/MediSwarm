# Speaker notes: ODELIA consortium briefing, Bremen, 17.09.2026

Spoken, informal. About one to two minutes per slide, 27 slides in 45 minutes.

## 1. Month 45

Hi everyone. I'm Jeff, I run the swarm platform for the consortium from Dresden, and this is where we stand at month 45. Four numbers to start: 34 thousand training volumes across eight hospitals, none of which ever left its site. A shared model at 0.887 malignant AUROC on the external challenge set, and the same number on an American cohort it has never seen. And software version 1.8.1, which went out to all of you this week.

## 2. Agenda

Five parts. First the data, what each of you brought. Then what the model does with it, including the external validation. Then the other deliverables, and I'll explain each one before I show numbers, because some of them are new to most of you. Then the milestone timeline and what we can close this year. And at the end, where each site stands and three things I need from you.

## 3. Training data per site

This is the training data per site, counted by each site's own trainer when it loads the data set, so these are the real numbers, not what was promised. Aachen alone is half of the consortium. The smallest site has 190 volumes. That 94-fold range shapes almost every technical decision I'll show you today.

## 4. How the data set grew

How we got there. Two sites in April, and then two big jumps: Aachen and Cambridge together on the eleventh of June, Utrecht in July. The eighth site came online at the end of July, and since then the data set has been stable.

## 5. Class counts per site

Same sites, now split by class at true scale. The thing to notice is the light blue: the benign cases. Aachen holds 84 percent of all benign cases in the consortium. VHIO has none. Keep that in mind for the robustness slides later, it comes back.

## 6. Class distribution per site

And the same thing normalised, so you can see each site's own mix. RSH looks very malignant-heavy at 63 percent, but that is 221 volumes. Aachen's 7 percent malignant is 1,251 volumes. So please read this slide together with the previous one, not on its own.

## 7. Site data detailed view

All the numbers in one place, for reference. The classes add up per site and the sites add up to 34,462. One correction from an earlier version: four join dates were about six weeks too early, because our own test machines had reported under hospital names. Test sites have their own names now.

## 8. Swarm model versus single-site models

Now the result. The eight-site swarm model on the external challenge set, against the previous six-site swarm and against single-site models. The swarm beats every single-site model, including Aachen's, and Aachen trains on half the data. For a typical site, joining is worth about 0.23 AUROC.

## 9. Results with confidence intervals

Same numbers with their uncertainty. The challenge set has only 37 malignant cases, so an AUROC on it is known to about plus or minus 0.06. That means 0.887 and 0.903 are within one interval of each other, and I would not argue about the third decimal.

## 10. External validation on the Duke cohort

This is the slide I'm most pleased with. Duke is a public American data set, 260 volumes, none of them anywhere near our training. Same task, malignant against no lesion. We lose 0.016 and the intervals overlap. A model trained across eight European hospitals transfers to a US cohort without a measurable loss.

## 11. Operating point: recall and specificity

One caveat before anyone takes the model into a clinic. AUROC measures ranking. The threshold is a separate choice, and ours is conservative: on the challenge set we catch about half the cancers and almost never raise a false alarm. On Duke the same picture. Moving that threshold costs nothing, but where it should sit is a clinical question, and I'll come back to it at the end.

## 12. Accuracy versus AUROC per site

Since 1.8 every site's metrics come back to the coordinator with their class counts, and this is why that matters. RUMC reports 99 percent accuracy. 888 of its 896 validation cases have no lesion, so a model that always says no lesion scores 99 percent and finds no cancer. The AUROC of 0.42 tells the truth. And UMCU was once called weak on a macro average where one class had two cases. Drop that class and it moves from 0.60 to 0.72. So: no ranking of sites on averages over tiny classes.

## 13. The remaining WP2 / WP3 deliverables

Now the other deliverables. This table is the map: what the proposal asks for, when it is due, and which of you I need for it. Three are due in December, three in June next year, one at the end of the project. I'll take them one by one, and for each I first explain what it is, then show what exists.

## 14. Regional fine-tuning: what is needed and what exists

Regional fine-tuning. The proposal wants the pan-European model compared with versions fine-tuned to the Dutch and the Greek cohorts. What I need for that is per-case predictions coming back from RUMC, UMCU and MHA: a row number, the label and three probabilities. No identifiers, no images. That feature is in 1.8, verified on real kits, and it is off by default: each site switches it on. The comparison itself happens in the October run.

## 15. Active learning: how it works

Active learning. The idea: labelling breast MRI is the expensive part, so let the model tell the radiologist which cases are worth labelling next. The model scores every unlabelled case by how uncertain it is, we take the top k, a radiologist labels them, they go into the training set, retrain, repeat. The control is the dashed path: pick k cases at random with the same budget. If the model can't beat random, it isn't useful.

## 16. Active learning: acquisition result

And it does beat random. With a budget of 10 cases, entropy selection finds 5 malignant cases where random finds 2 or 3. At 20, 9 against 4 or 5. So roughly double, where the budget is small, which is exactly where it matters. By 100 cases everyone has taken most of the pool and the methods converge, as they must. What this doesn't show yet is a model retrained on those cases. That's the reduced retraining run in October and November.

## 17. Differential privacy: where the noise goes

Differential privacy. What the proposal asks is optional noise in training with a privacy budget. Here is where the noise goes. A site trains, produces a model update. We clip that update to a fixed norm so no single site can push too hard, then add Gaussian noise, and only then send it. An accountant adds up the privacy cost, epsilon, over the rounds and stops training when the budget is spent. The guarantee is at the hospital level: from the shared model you cannot tell whether a given hospital took part in a round.

## 18. Differential privacy: guarantee and cost

So what does a guarantee cost? This table is for a 20-round run. Epsilon of 10, which is the commonly accepted level, needs noise scale 2.4. Epsilon 3 needs 6.7. Epsilon 1 would need 18 and probably destroys the model. The honest gap: we have not yet measured what 2.4 costs in accuracy. One 20-round run does that, and the noise level is then a decision for this room, not a default I set.

## 19. Adversarial robustness: attack and defence

Adversarial robustness. In a swarm every site sends weights, so every site is a potential attacker. Here eight sites send updates, one of them poisoned. With the aggregation rule we use today, a plain weighted mean, there is no bound: VHIO with 0.6 percent of the data can move the shared model by five and a half times its share. With a norm-bounded mean, each contribution is capped, every site stays in, and the same attack moves the model by 0.04.

## 20. Adversarial robustness: measured effect of five rules

We tried five rules. The textbook robust ones, trimmed mean, coordinate median, Krum, cost a lot even with no attacker at all, 0.47 to 0.64 error. That's because they throw away contributions and ignore data-set weights, so with Aachen holding 84 percent of benign cases they aim at the average site instead of the pooled data. The norm-bounded mean costs almost nothing and holds the attack. That's the recommendation. It does not limit the largest contributor, and capping any site's share is a governance decision, not mine.

## 21. White-hat attack: plan and interim work

The white-hat attack, milestone 6. In the proposal, Cambridge and Nijmegen attack the trained model to expose vulnerabilities, and the deliverable is a preprint. Top row is that plan: we give you the model weights and a protocol, you try to reconstruct data or infer membership, we write up the findings and the fix together. What I need from Cambridge and Nijmegen is a slot in the first quarter of next year, and I'd like to fix the date this autumn.

## 22. White-hat attack: result of the interim probe

In the meantime we ran the part that needs nobody else. The question: from a case's three output probabilities alone, can you tell which hospital it came from? On no-lesion cases, no. On malignant cases, yes: 60 percent correct against 33 percent chance. So the model behaves differently per site exactly on the clinically relevant cases. This is a lower bound, the real attacker has the weights. But it tells Cambridge and Nijmegen where to look first, and it is relevant for the per-case predictions I'm asking for.

## 23. Grant tasks against the calendar

The timeline. One bar per deliverable, from the grant's task window to the due date, filled to how much of its checklist is done. The orange line is today, the dashed one is the end of the year. The three December items are between 50 and 60 percent. The June items are earlier, which is fine, they have nine months.

## 24. What concludes by 31 December 2026

The checklists behind those percentages, so you can hold me to them. On the left, the three that must close this year: regional fine-tuning, active learning, robustness. On the right, what continues into 2027. And the plan: kits this month, the big benchmark run in October, the extra experiments in October and November, drafts in November, submission in December. The one thing that decides the year is the October run: all eight sites online for two days.

## 25. Deliverable status, 15 September 2026

Status in one table. And a short note on what happened since the software went out: 1.8.1 is verified on real kits, MHA already installed it, Cambridge's SAM-Med2D model is merged into the platform, and a four-site fault-injection test last weekend found two defects that are already fixed and in review. One of them, a worker crash at start-up, had actually hit Cambridge once in April, so that one matters for the October run.

## 26. Site status and to-do list, 15 September 2026

Where each site stands as of Monday. Everyone is connected to the coordinator. MHA is on 1.8.1, thank you. The others still need to install the kit, it's ten minutes and the certificates don't change. RSH and USZ are on the old 1.5 kit that cannot follow releases, so for you the kit is the only way. RUMC and USZ, your log feed has been silent for a while, please check the upload key. UMCU, you re-registered on Sunday but the feed stopped, which usually means the sync file wasn't copied into the new kit.

## 27. Three requests to the consortium

Three requests. One: RUMC, UMCU and MHA, switch on per-case return before the October run. You saw on slide 22 that those rows carry some site information, so decide with that in mind. Two: Cambridge and Nijmegen, a slot for the white-hat exercise in the first quarter. Three: a clinical view on the operating point. The model misses half the cancers at a 1 percent false-alarm rate. Moving the threshold is free; deciding where it goes is yours. Thank you.
