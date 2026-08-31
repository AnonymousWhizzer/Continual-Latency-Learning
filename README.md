## Results Overview

The experiments compare three prediction strategies: **Baseline**, **Fixed**, and **Adaptive**.

### Baseline

The baseline relies on a single model. Its predictions become highly unstable around regime changes, with strong oscillations after each breakpoint.

This behavior illustrates the limitations of a single continually updated model, including **catastrophic forgetting** and **remembering effects**: adapting to a new regime can degrade previously learned knowledge, while returning to an older regime may reactivate unstable prediction behavior.

### Fixed model pool

The fixed approach improves over the baseline by selecting among several pre-trained models.

It reduces long-term instability, but prediction errors are still important around regime changes. Because the model pool is static, the system cannot create a new model when an unseen regime appears.

### Adaptive model pool

At **Event 1**, the adaptive approach behaves similarly to the fixed approach because the new QoS regime has not yet been learned. This is why a QoE degradation is still observed during Event 1.

This first exposure triggers retraining and leads to the creation and integration of a new specialized model, `PoP_4248 (ESN)`.

After this adaptation phase, the behavior changes significantly:

- **Event 1:** Fixed ≈ Adaptive → new regime not yet learned
- **Retraining:** a new model is created and added to the adaptive pool
- **Event 2:** Adaptive reuses the learned regime and remains stable
- **Event 3:** Adaptive again handles the regime without significant prediction instability

The fixed approach, in contrast, continues to exhibit prediction oscillations because its model pool cannot evolve.

## QoE Impact

The QoE results confirm that the prediction improvements translate into application-level benefits.

During **Event 1**, both Fixed and Adaptive experience a strong playback slowdown and rebuffering because the adaptive system has not yet learned the new regime.

After retraining, the difference becomes clear:

- Fixed still experiences severe slowdown and playback interruptions during Events 2 and 3.
- Adaptive maintains playback close to its nominal behavior.
- No rebuffering is observed for Adaptive during Events 2 and 3.

Importantly, the network degradation is still injected during these events. What disappears is its **impact on the application-level QoE**.

Overall, the results show that the main benefit of continual adaptation is not only lower prediction error, but improved **prediction reliability around regime changes**, which directly translates into better service continuity.