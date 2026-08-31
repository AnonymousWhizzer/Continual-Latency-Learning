Experimental Results

This section presents the comparison between the Baseline, Fixed, and Adaptive prediction strategies, with a particular focus on prediction stability around regime changes and the resulting impact on application-level QoE.

1. Baseline: prediction instability and forgetting effects

The Baseline relies on a single prediction model.

The results show strong prediction oscillations around the detected regime changes. Each transition introduces a period of instability before the predictor converges toward the new RTT level.

This behavior highlights the limitations of a single continually updated model in a non-stationary environment. In particular, the repeated instability observed when the operating conditions change is consistent with catastrophic forgetting and remembering effects: adapting the same model to a new regime can degrade previously acquired knowledge, while returning to a previously observed regime does not necessarily guarantee an immediate stable response.

<p align="center">
  <img src="figures/Baseline_all_events.png" width="95%" alt="Baseline prediction behavior">
</p>

The Baseline therefore performs reasonably during stationary periods, but its prediction reliability deteriorates significantly around regime transitions.

2. Fixed model pool: improved stability but persistent transition errors

The Fixed strategy improves over the Baseline by selecting among a fixed set of pre-trained models.

This reduces the need to continuously overwrite the knowledge of a single model and therefore limits the instability observed with the Baseline.

However, prediction errors remain visible around the breakpoints. The model pool is static, meaning that the system can only select among models that already exist. When a newly observed QoS regime is not sufficiently represented by the available models, the Fixed strategy cannot create a more appropriate predictor.

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

Compared with the Baseline, the Fixed approach is more stable overall, but significant prediction oscillations can still occur immediately after regime changes.

3. Adaptive model pool: learning a new regime

The Adaptive strategy extends the Fixed approach by allowing the model pool to evolve when the existing models are no longer sufficient.

Event 1: behavior similar to Fixed

At Event 1, the Adaptive strategy behaves similarly to Fixed.

At this stage, the newly encountered QoS regime has not yet been learned and no specialized model exists for it. The prediction therefore remains unstable around the corresponding breakpoint.

<p align="center">
  <img src="figures/adaptive_event_1.png" width="90%" alt="Adaptive approach - Event 1">
</p>

This first exposure triggers the retraining mechanism.

A new specialized model, PoP_4248 (ESN), is then created and integrated into the Adaptive model pool.

Events 2 and 3: reuse of the learned regime

Once the new model has been integrated, the Adaptive strategy can reuse the knowledge acquired during Event 1.

When the corresponding QoS regime is encountered again, the predictor no longer needs to adapt from scratch. This results in a much more stable prediction response around the following regime changes.

<p align="center">
  <img src="figures/adaptive_event_3.png" width="90%" alt="Adaptive approach after retraining">
</p>

The sequence can therefore be summarized as:

Event 1
   ↓
New QoS regime encountered
   ↓
Fixed ≈ Adaptive
   ↓
Prediction instability
   ↓
Retraining
   ↓
New specialized model created
   ↓
Model integrated into the Adaptive pool
   ↓
Events 2 and 3
   ↓
Previously learned regime is reused
   ↓
More stable predictions

The key advantage of the Adaptive strategy is therefore not only model selection, but also the ability to extend the model pool when a previously unseen operating condition appears.

QoE Impact

The prediction-level results are confirmed by the application-level QoE measurements.

The experiments introduce controlled network degradations and observe their impact on playback slowdown, buffering, rebuffering, and recovery.

QoE Event 1

During Event 1, Fixed and Adaptive exhibit a similar behavior.

This is expected because the Adaptive strategy has not yet learned the new regime. The disturbance therefore produces a significant playback degradation for both strategies.

<p align="center">
  <img src="figures/qoe_event_1.png" width="95%" alt="QoE Event 1">
</p>

Event 1 can therefore be interpreted as the adaptation phase: the new condition exposes the limitation of the existing model pool and triggers retraining.

QoE Event 2

After retraining and integration of the new model, the difference between Fixed and Adaptive becomes clear.

The Fixed strategy still experiences a strong playback slowdown and rebuffering, while the Adaptive strategy is able to reuse the newly learned regime.

<p align="center">
  <img src="figures/qoe_event_2.png" width="95%" alt="QoE Event 2">
</p>

For Adaptive, the network degradation is still injected, but it no longer produces the same application-level QoE degradation.

QoE Event 3

The same trend is confirmed during Event 3.

<p align="center">
  <img src="figures/qoe_event_3.png" width="95%" alt="QoE Event 3">
</p>

The Fixed strategy still exhibits severe slowdown and playback interruption.

Adaptive, on the other hand, reuses the knowledge acquired after Event 1 and avoids the QoE degradation observed with Fixed.

No rebuffering is observed for Adaptive during Events 2 and 3.

Main Observation

The experiments highlight three distinct behaviors:

Strategy

Prediction behavior around regime changes

Model pool

QoE impact

Baseline

Strong instability and repeated oscillations

Single model

High sensitivity to degradation

Fixed

More stable than Baseline, but still unstable around breakpoints

Static pool

Rebuffering remains observable

Adaptive

Similar to Fixed at first exposure, then significantly more stable

Evolvable pool

No observed rebuffering during Events 2 and 3

The main benefit of the Adaptive strategy is therefore not limited to a reduction in average prediction error.

Its main advantage is the improvement of prediction reliability around regime changes.

The first occurrence of a new regime may still generate prediction instability and QoE degradation. However, after retraining and integration of a specialized model, subsequent occurrences of the same or a similar regime can be handled more effectively.

This improvement directly translates into better service continuity and application-level QoE.
