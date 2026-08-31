Experimental Results

This document gives additional insights by comparing three QoS prediction strategies:

* Baseline: a single prediction model
* Fixed: a fixed pool of pre-trained models
* Adaptive: an evolving model pool able to integrate a new specialized model when a previously unseen QoS regime is encountered.

The analysis focuses on two aspects:
  * prediction stability around regime changes;
  * the resulting impact on application-level QoE.

Prediction Results

1. Baseline: prediction instability and forgetting effects

The Baseline relies on a single prediction model.

The results show strong prediction oscillations around regime changes. After each breakpoint, the model requires several prediction windows before converging toward the new RTT level.

This behavior highlights the limitations of a single continually updated model in a non-stationary environment. In particular, the repeated instability observed when the operating conditions change is consistent with catastrophic forgetting and remembering effects.

When the model adapts to a new regime, previously acquired knowledge may be degraded. Conversely, when a previously observed regime reappears, the model does not necessarily recover a stable prediction behavior immediately.

<p align="center">
  <img src="figures/Baseline_all_events.png" width="95%" alt="Baseline prediction behavior">
</p>

The Baseline therefore performs reasonably during stationary periods, but its prediction reliability deteriorates significantly around regime transitions.

2. Fixed model pool: improved stability but persistent transition errors

The Fixed strategy improves over the Baseline by selecting among a fixed set of pre-trained models.

Instead of continuously modifying a single model, the system can switch between specialized predictors according to the current QoS conditions.

This reduces the long-term instability observed with the Baseline and avoids continuously overwriting the knowledge of a single predictor.

However, the model pool remains static. The system can only select among models that already exist. When the observed QoS regime is not sufficiently represented by the available models, prediction instability remains visible around the breakpoints.

Event 1

<p align="center">
  <img src="figures/fixed_event_1.png" width="90%" alt="Fixed approach - Event 1">
</p>

Event 2

<p align="center">
  <img src="figures/fixed_event_2.png" width="90%" alt="Fixed approach - Event 2">
</p>

Event 3

<p align="center">
  <img src="figures/fixed_event_3.png" width="90%" alt="Fixed approach - Event 3">
</p>

Compared with the Baseline, the Fixed approach is more stable overall. Nevertheless, significant prediction oscillations remain immediately after some regime changes because the model pool cannot evolve when a new operating condition appears.

3. Adaptive model pool: learning a new regime

The Adaptive strategy extends the Fixed approach by allowing the model pool itself to evolve.

When the currently available models are not sufficient to represent a newly observed QoS regime, the system can trigger retraining and integrate a new specialized model.

Event 1: first exposure to the new regime

During Event 1, Adaptive behaves similarly to Fixed.

At this stage, the newly encountered QoS regime has not yet been learned and no specialized model is available for it. Prediction instability therefore remains visible around the corresponding breakpoint.

<p align="center">
  <img src="figures/adaptive_event_1.png" width="90%" alt="Adaptive approach - Event 1">
</p>

This first exposure triggers the retraining mechanism.

A new specialized model, PoP_4248 (ESN), is then created and integrated into the Adaptive model pool.

Events 2 and 3: reuse of the learned regime

Once the new model has been integrated, Adaptive can reuse the knowledge acquired during Event 1.

When the corresponding or a sufficiently similar QoS regime is encountered again, the predictor no longer needs to adapt from scratch. The prediction response becomes considerably more stable around the following regime changes.

<p align="center">
  <img src="figures/adaptive_event_3.png" width="90%" alt="Adaptive approach after retraining">
</p>

Adaptive learning process

Stage

System behavior

Consequence

Event 1

A new QoS regime is encountered. Fixed and Adaptive behave similarly because no specialized model is available yet.

Prediction instability is observed.

Retraining

The new regime triggers the retraining process.

A new specialized model, PoP_4248 (ESN), is created.

Model integration

The new model is added to the Adaptive model pool.

The system now has a representation of the previously unseen regime.

Event 2

A similar regime is encountered again.

Adaptive reuses the learned model and produces more stable predictions.

Event 3

The learned regime appears again.

Adaptive remains stable and avoids the degradation observed with Fixed.

This sequence highlights an important property of the Adaptive strategy: its advantage does not necessarily appear during the first encounter with a new regime.

The benefit emerges after the system has learned from this first exposure and integrated a specialized model into its pool.

QoE Impact

The prediction-level results are confirmed by the application-level QoE measurements.

The experiments introduce controlled network degradations and observe their impact on:

playback slowdown;

buffering;

rebuffering;

service recovery.

Baseline

The Baseline is the most affected by the network disturbances.

The playback interruptions are particularly severe. Once the stream enters a strong rebuffering state, it is unable to recover automatically before the end of the degradation period.

A manual restart is therefore required to resume the stream.

This behavior highlights the poor resilience of the single-model approach under successive regime changes and is consistent with the strong prediction instability observed around the breakpoints.

<p align="center">
  <img src="figures/qoe_event_1.png" width="95%" alt="QoE comparison - Event 1">
</p>

The Baseline therefore represents the worst-case behavior: prediction instability is accompanied by severe application-level interruption and manual intervention is required to restore playback.

QoE Event 1

During Event 1, Fixed and Adaptive exhibit a similar behavior.

This is expected because Adaptive has not yet learned the new regime. The newly observed QoS condition is not represented by a dedicated model in the Adaptive pool.

As a consequence, both Fixed and Adaptive experience a strong playback slowdown and rebuffering.

<p align="center">
  <img src="figures/qoe_event_1.png" width="95%" alt="QoE Event 1">
</p>

Unlike the Baseline, however, Fixed and Adaptive are able to recover without requiring a manual restart.

Event 1 therefore represents the adaptation phase for Adaptive:

the new QoS condition is encountered;

the current model pool proves insufficient;

QoE degradation is observed;

retraining is triggered;

a new specialized model is learned.

QoE Event 2

After retraining and integration of the new model, the difference between Fixed and Adaptive becomes clear.

The Fixed strategy continues to experience a strong playback slowdown and rebuffering because its model pool is unchanged.

Adaptive, on the other hand, can reuse the model learned after Event 1.

<p align="center">
  <img src="figures/qoe_event_2.png" width="95%" alt="QoE Event 2">
</p>

The network degradation is still injected during Event 2, but Adaptive no longer exhibits the same application-level degradation.

No rebuffering is observed for Adaptive.

QoE Event 3

The same trend is confirmed during Event 3.

<p align="center">
  <img src="figures/qoe_event_3.png" width="95%" alt="QoE Event 3">
</p>

The Fixed strategy again exhibits severe playback slowdown and interruption.

Adaptive reuses the knowledge acquired after Event 1 and maintains playback close to its nominal behavior.

No rebuffering is observed for Adaptive during Event 3.

From Prediction Stability to QoE

The QoE results reinforce the prediction-level observations.

Baseline

The single-model strategy exhibits strong instability around regime changes. This instability translates into severe playback interruptions, to the point that a manual restart is required to continue the stream.

Fixed

The fixed model pool improves prediction stability and service recovery compared with the Baseline.

However, because its model pool cannot evolve, prediction instability remains around some breakpoints and the application still experiences significant rebuffering.

Adaptive

At Event 1, Adaptive behaves similarly to Fixed because the new regime has not yet been learned.

After retraining and integration of PoP_4248 (ESN), Adaptive can recognize and reuse the learned regime during Events 2 and 3.

As a result:

prediction instability is strongly reduced;

playback slowdown remains close to its nominal behavior;

no rebuffering is observed during Events 2 and 3.

Importantly, the network degradation is still injected during these events. What disappears is not the network disturbance itself, but its impact on application-level QoE.

Main Observation

The experiments highlight three distinct behaviors:

Strategy

Prediction behavior around regime changes

Model pool

Application-level impact

Baseline

Strong instability, catastrophic forgetting and remembering effects

Single model

Severe interruptions; manual restart required

Fixed

More stable than Baseline, but oscillations remain around breakpoints

Static model pool

Automatic recovery, but rebuffering remains

Adaptive

Similar to Fixed during the first exposure, then significantly more stable

Evolvable model pool

No observed rebuffering during Events 2 and 3

The main benefit of the Adaptive strategy is therefore not limited to reducing average prediction error.

Its main advantage is the improvement of prediction reliability around regime changes.

The first occurrence of a new QoS regime can still generate prediction instability and QoE degradation. However, this first exposure allows the system to learn the new operating condition.

After retraining and integration of a specialized model, subsequent occurrences can be handled much more effectively.

In this experiment:

Event 1 exposes the new regime and triggers adaptation;

retraining creates and integrates PoP_4248 (ESN);

Events 2 and 3 demonstrate the benefit of the acquired knowledge;

the improved prediction stability directly translates into better service continuity and application-level QoE.
