# Speaker notes: ODELIA consortium briefing, Bremen, 17.09.2026

Spoken, informal. About one to two minutes per slide, 27 slides in 45 minutes.

## 1. Month 45

Hi everyone. I'm Jeff, I run the swarm platform for the consortium at TUD, and this is where we stand at month 45. Four numbers to start: 34 thousand training volumes across eight hospitals, none of which ever left its site. A shared model at 0.887 malignant AUROC on the external challenge set, and the same number on an American cohort it has never seen. And software version 1.8.1, which went out to all of you on Monday.

## 2. Agenda

Five parts. First the data, what each of you brought. Then what the model does with it, including the external validation. Then the other deliverables, and I'll explain each one before I show numbers, because some of them are new to most of you. Then the milestone timeline and what we can close this year. And at the end, where each site stands and three things I need from you.

## 3. Training data per site

This is the training data per site, counted by each site's own trainer when it loads the data set, so these are the real numbers, not what was promised. UKA alone is half of the consortium. The smallest site has 190 volumes. That 94-fold range shapes almost every technical decision I'll show you today.

## 4. How the data set grew

How we got there. Two sites in April, and then two big jumps: UKA and CAM together on the eleventh of June, UMCU in July. The eighth site came online at the end of July, and since then the data set has been stable.

## 5. Class counts per site

Same sites, now split by class at true scale. The thing to notice is the light blue: the benign cases. UKA holds 84 percent of all benign cases in the consortium. VHIO has none. Keep that in mind for the robustness slides later, it comes back.

## 6. Class distribution per site

And the same thing normalised, so you can see each site's own mix. RSH looks very malignant-heavy at 63 percent, but that is 221 volumes. UKA's 7 percent malignant is 1,251 volumes. So please read this slide together with the previous one, not on its own.

## 7. Site data detailed view

All the numbers in one place, for reference. The classes add up per site and the sites add up to 34,462. One correction from an earlier version: four join dates were about six weeks too early, because our own test machines had reported under hospital names. Test sites have their own names now.

## 8. Swarm model versus single-site models

Now the result. The eight-site swarm model on the external challenge set, against the previous six-site swarm and against single-site models. The swarm beats every single-site model, including UKA's, and UKA trains on half the data. For a typical site, joining is worth about 0.23 AUROC.

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

Regional fine-tuning. The proposal wants the consortium model compared with versions fine-tuned to the RUMC and UMCU cohort and to the MHA cohort. What I need for that is per-case predictions coming back from RUMC, UMCU and MHA: a row number, the label and three probabilities. No identifiers, no images. That feature is in 1.8, verified on real kits, and it is off by default: each site switches it on. The comparison itself happens in the October run.

## 15. Active learning: how it works

Active learning. The problem this deliverable solves is simple: the expensive part of our work is not the training, it is the labelling. Every labelled case costs a radiologist's time. So if a radiologist gives us time for twenty more cases, which twenty should we ask for? The naive answer is any twenty. Active learning says: ask the model. Follow the diagram from left to right. On the left is a pool of cases with images but no label. The model gives each case three numbers: the probability of no lesion, of benign, of malignant. For an easy case those look like 95, 3 and 2 percent. For a hard case they look like 40, 30 and 30. The second one is the case the model is unsure about, and that is what the middle box measures. Entropy just means how spread out the three numbers are; high entropy, the model is unsure. We sort all cases by that number and take the top k. That k is the labelling budget, the number of cases the radiologist has time for. The radiologist labels them, they go into the training set, we retrain, and the loop starts again. The model chooses its own homework. The dashed path at the bottom is the control: same budget, same radiologist, but the k cases are drawn at random. Any method has to beat that. One more thing, which is why the next slide counts malignant cases: on the challenge set the model calls 21 cases malignant while 37 truly are. It under-calls cancer, so malignant cases are the ones it needs most.

## 16. Active learning: acquisition result

This table is the selection step alone, before any retraining. We took the 165 challenge cases, hid their labels and treated them as the unlabelled pool; 37 of them are malignant, about 22 percent. The budget is simply how many cases you are allowed to have labelled. Both methods pick that many, we uncover the labels of the picked cases and count the malignant ones. Random column: draw the budget at random, count, repeat many times, report the average and the spread. Since 22 percent of the pool is malignant, ten random cases contain about two malignant ones; that column is just budget times 22 percent. Entropy column: the model scores all 165 cases once, sorts them by how unsure it is, and hands over the top of the list. The ten most uncertain cases contained 5 malignant ones; there is no plus or minus because there is no randomness. Budget of 20: random gets 4 or 5, the model gets 9. At 30 and 40 the ratio is still 1.6. So at every small budget, which is the realistic budget, the model finds roughly twice as many of the cases it needs. The last row is a sanity check: with 100 of 165 cases picked, both methods have taken most of the 37 malignant cases, so they meet at 21 and 22. The advantage lives where budgets are small, and that is where we are. What this does not show yet is the second half: that a model retrained on the selected cases gets better than one retrained on the random ones. That is the reduced retraining run in October and November, two arms at two budgets, about 45 hours of GPU across the fleet. When that is in, D3.2 is complete.

## 17. Differential privacy: where the noise goes

Differential privacy is a formal way of saying: what leaves a hospital must not reveal too much about what is inside it. Not a policy, a mathematical guarantee with a number attached. The proposal asks for two things: optional noise in training, and a privacy budget that the software enforces. The diagram follows one site through one round. The site trains and produces a model update, a long list of numbers saying how the shared model should change. Two things happen before it leaves. First, clipping: we cap the size of the update, so no single site can push the shared model further than a fixed amount. Second, noise: we add random numbers calibrated to that cap, large enough that nobody can work backwards from the update to the data, small enough that, averaged over eight sites and twenty rounds, the model still learns. Only that clipped, noised update travels. The images never leave. The box underneath is the accountant. Every noisy round spends a bit of privacy, and the accountant adds it up as a number called epsilon; smaller epsilon, stronger privacy. When the agreed budget is spent, training stops. That is the budget by design. The guarantee is at the hospital level: someone who sees the shared model cannot tell whether a given hospital took part in a round. Clipping and noise are in the software since 1.8. Still missing: the accountant writing epsilon into the metrics, the automatic stop, and the measurement of what the noise costs in accuracy.

## 18. Differential privacy: guarantee and cost

This table answers one question: how much noise do we have to add to get a given guarantee? The left column is the guarantee, epsilon, for a full twenty-round run. The middle column, sigma, is the amount of noise relative to the clipping cap. Bigger sigma, more noise, stronger privacy, and a worse model. Epsilon of 10 is what most published medical work uses; it needs a sigma of 2.4. Epsilon of 3 is a strong guarantee; it needs 6.7, almost three times as much noise. Epsilon of 1 is the strictest number people cite; it needs 18, and at our scale of eight sites that will most likely destroy the model. The small delta in the caption is a standard technicality, an allowed failure probability of one in a hundred thousand. Two honest limits. The guarantee covers whether a hospital took part. It does not cover whether a particular patient was in a hospital's data; that needs noise inside the local training itself, and we do not claim it. And we have not yet measured what sigma 2.4 costs in accuracy. One twenty-round run with the noise on gives us that number, and then this table becomes a real trade-off. In one line: a hospital-level guarantee at the commonly accepted level is within reach at sigma 2.4, but its cost in accuracy is unmeasured, so the consortium should choose epsilon only after that run.

## 19. Adversarial robustness: attack and defence

Adversarial robustness. The proposal asks for a report on how a swarm can be attacked and what protects it. The uncomfortable starting point: in a swarm every site sends model weights to whoever aggregates the round, and nobody inspects them. So every site is a potential attacker, by intent, by a bug, or through a compromised machine that sends garbage. The diagram shows one round. The eight sites send their updates, weighted by how much data they hold: UKA 52 percent, UMCU 18, down to VHIO with 0.6. One update is poisoned, and I chose the smallest site on purpose, to show that size does not protect us. The rule we use today is a plain weighted mean with no limit on how large any single update may be. If VHIO sends an update ten times too large, the mean simply takes it: the shared model moves 5.5 times VHIO's actual share. The defence at the bottom is the norm-bounded mean. Before averaging, each site's update is capped at a length proportional to that site's weight; a too-large update is shrunk to the cap, a normal one passes unchanged. Every site stays in, the data-set weights stay in. Same attack: the model moves by 0.04, and with no attack at all by 0.032. The attack has almost stopped mattering.

## 20. Adversarial robustness: measured effect of five rules

We tried five aggregation rules on a simulation with the consortium's real data-set weights. Two columns matter: the error with no attacker, which is what the rule costs us in a normal round, measured as the distance from the correct weighted average; and what happens under the attack. The weighted mean, our current rule, costs nothing but offers no protection. The norm-bounded mean costs 0.032, almost nothing, and holds the attack at 0.04. Those are the two rows to compare. The three rules below are the textbook answers. Trimmed mean throws away the largest and smallest values, coordinate median takes the middle value, Krum keeps only the one update that looks most like the others. They all bound an attacker, but look at the first column: 0.47, 0.60, 0.64. A huge error with nobody attacking, because these rules treat every site as equal and throw information away. Our sites are anything but equal: UKA holds 84 percent of all benign cases. A rule that ignores the weights aims at the average site instead of the pooled data, and that alone costs more than the attack it prevents. So the recommendation is the norm-bounded mean: cap every contribution, keep every site, keep the weights. One limit to state openly: the cap is proportional to weight, so it does not limit UKA. Protecting against the largest site too would mean capping its share, which costs accuracy; that is a governance decision for the consortium. Still open for the report: a dishonest aggregating site, which we simulate in October.

## 21. White-hat attack: plan and interim work

The white-hat attack, milestone 6. In the proposal, CAM and RUMC attack the trained model to expose vulnerabilities, and the deliverable is a preprint. Top row is that plan: we give you the model weights and a protocol, you try to reconstruct data or infer membership, we write up the findings and the fix together. What I need from CAM and RUMC is a slot in the first quarter of next year, and I'd like to fix the date this autumn.

## 22. White-hat attack: result of the interim probe

In the meantime we ran the part that needs nobody else. The question: from a case's three output probabilities alone, can you tell which hospital it came from? On no-lesion cases, no. On malignant cases, yes: 60 percent correct against 33 percent chance. So the model behaves differently per site exactly on the clinically relevant cases. This is a lower bound, the real attacker has the weights. But it tells CAM and RUMC where to look first, and it is relevant for the per-case predictions I'm asking for.

## 23. Grant tasks against the calendar

The timeline. One bar per deliverable, from the grant's task window to the due date, filled to how much of its checklist is done. The orange line is today, the dashed one is the end of the year. The three December items are between 50 and 60 percent. The June items are earlier, which is fine, they have nine months.

## 24. What concludes by 31 December 2026

The checklists behind those percentages, so you can hold me to them. On the left, the three that must close this year: regional fine-tuning, active learning, robustness. On the right, what continues into 2027. And the plan: kits this month, the big benchmark run in October, the extra experiments in October and November, drafts in November, submission in December. The one thing that decides the year is the October run: all eight sites online for two days.

## 25. Deliverable status, 17 September 2026

Status in one table. And a short note on what happened since the software went out: 1.8.1 is verified on real kits, MHA, CAM and USZ have their kits ready, CAM's SAM-Med2D model is merged into the platform, and fault-injection tests over the last three days found three defects. One, a worker crash at start-up, had actually hit CAM once in April; that fix is merged and matters for the October run. The second, the tolerant mode not telling the other sites when one is dropped, is fixed and passed its test yesterday. The third is a small timing race that I've filed.

## 26. Site status and to-do list, 17 September 2026

Where each site stands as of this morning. Everyone is connected to the coordinator. MHA, CAM and USZ, thank you, your emails came in this morning: the 1.8.1 kits are ready at all three, and CAM's log feed already comes from the new kit. RSH asked for a hand with the setup, so we'll do it together in a shared session this week and stop the three old kits at the same time. UKA, RUMC and VHIO still need to install the kit; it's ten minutes and the certificates don't change. One thing I saw on Tuesday evening: UMCU has two clients running under the same identity, so the second one keeps re-registering every few seconds and gets rejected; stop the old one and keep the 1.8.1 kit. RUMC and USZ, your log feed has been silent for a while, please check the upload key. USZ, once the feed is back I can confirm the kit from here.

## 27. Three requests to the consortium

Three requests. One: RUMC, UMCU and MHA, switch on per-case return before the October run. You saw on slide 22 that those rows carry some site information, so decide with that in mind. Two: CAM and RUMC, a slot for the white-hat exercise in the first quarter. Three: a clinical view on the operating point. The model misses half the cancers at a 1 percent false-alarm rate. Moving the threshold is free; deciding where it goes is yours. Thank you.
