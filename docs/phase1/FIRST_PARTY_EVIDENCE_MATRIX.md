# Phase 1 first-party evidence matrix

Status: incremental Phase 1 evidence record. Entries document what a primary source does and does not support; they are not empirical evidence for the candidate method and do not establish novelty by themselves.

Technical claims in this matrix use original papers, official proceedings, author/institution project pages, or official code repositories. Search-engine snippets and third-party summaries are discovery aids only and are not evidence entries.

## Recency audit status

The 2026-08-22 recency search and subsequent first-party full-text collection completed the queued audits of MoRPI-PINN, the Sensors GNSS-IMU PINN/EKF paper, and the 2026 IEEE left-equivariant SINS/GNSS filter paper. Their admissible claims and limitations are recorded below. This closes the dated queue; it does not establish an indefinitely current novelty claim, and the search must be refreshed at the later novelty gate.

## EqNIO

**Work:** Royina Karegoudra Jayanth, Yinshuang Xu, Ziyun Wang, Evangelos Chatzipantazis, Kostas Daniilidis, and Daniel Gehrig, "EqNIO: Subequivariant Neural Inertial Odometry," ICLR 2025.

**Primary sources checked on 2026-08-21:**

- ICLR proceedings paper: <https://proceedings.iclr.cc/paper_files/paper/2025/file/6554c4151c2ccbd36da2d55c015b2039-Paper-Conference.pdf>
- ICLR proceedings record: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/6554c4151c2ccbd36da2d55c015b2039-Abstract-Conference.html>
- Author code/data page: <https://github.com/RoyinaJayanth/EqNIO>

**Verified method facts:**

- The learned odometry input is accelerometer and gyroscope data from one IMU, not GNSS.
- The TLIO form predicts a neural displacement prior and associated covariance for filter fusion; the RONIN form predicts an end-to-end velocity quantity.
- The structural contribution is a learnable canonical gravity-aligned frame and layers that commute with gravity-preserving roto-reflections. The paper evaluates strict structural equivariance against softer augmentation/loss approaches.
- The TLIO training description uses displacement computed from VIO position estimates as supervision. The RONIN training description derives target velocity from ground-truth position. The official repository states that the released processed training array includes VIO ground-truth data.

**Admissible influence on this project:**

- Supports typed gravity-preserving group actions, equivariant displacement/covariance outputs, the distinction between structural and soft equivariance, and the need to test the full output transformation.
- Supports beginning with the `SO(2)` subgroup while treating reflections only after every pseudovector parity rule is explicit.

**Not supported by this source:**

- It does not provide a GNSS/INS fusion objective.
- It does not provide training without trajectory supervision and is not a first-party precedent for this project's ESKF-pseudo-label plus inertial-physics objective.
- It does not show that equivariance alone resolves 20--30 second GNSS outages, GNSS measurement degradation, absolute position gauge, or teacher-label bias.

**Permitted project wording:** EqNIO is a structural antecedent for gravity-preserving equivariant neural inertial representations. It is not cited as a no-high-precision-supervision training solution.

## PINK-GINS

**Work:** Jianan Lou and Rong Zhang, "PINK-GINS: A Hybrid Physics-Informed Neural Network and Kalman Filter Framework for GNSS/INS Tightly Coupled Integration," ION GNSS+ 2025, pp. 1044-1054, DOI 10.33012/2025.20237.

**Primary sources checked on 2026-08-22:**

- User-supplied original proceedings PDF (local personal path intentionally omitted)
- ION official publication record: <https://www.ion.org/publications/abstract.cfm?articleID=20237>

**Verified method facts:**

- PINK-GINS is genuinely tightly coupled: its ESKF observation model uses corrected ionosphere-free code, carrier phase, and Doppler residuals per satellite, with receiver clock, wet-troposphere, and carrier-ambiguity states.
- Its dual branch consumes IMU windows and satellite quality/residual features. The network predicts accelerometer/gyroscope biases, process covariance `Q`, and per-satellite measurement covariance `R`; the ESKF remains the final navigation estimator.
- Equations (42)-(46) define discrete strapdown position, velocity, attitude, and rotation-orthogonality residuals and penalize them in a physics loss. Equation (53) adds a whitened innovation loss.
- Equation (54) explicitly supervises estimated state against `x_GT`; equation (55) includes that supervised term in the total loss. The experimental platform also includes a reference system for ground-truth collection.

**Admissible influence on this project:**

- Supports the relevance of explicit discrete inertial-mechanization residuals whose value depends on learned state/bias quantities.
- Supports treating per-satellite corrected GNSS observables and geometry as a true tightly coupled extension rather than relabelling PVT fusion as tight coupling.
- Provides structural comparison points for per-satellite covariance parameterization, innovation diagnostics, lever-arm-aware observation equations, and raw code/phase/Doppler modeling.

**Rejected or modified design choices:**

- Equation (54) is prohibited because high-precision trajectory truth may not affect this project's training, tuning, or checkpoint selection. The ESKF weak-pseudo-label term and independent inertial residual replace it; they are disclosed as weak supervision rather than called fully unsupervised.
- Joint free learning of biases, `Q`, and `R` is not copied because these quantities can compensate for one another and lose identifiability. This project fixes the process model and does not learn a free loose-coupling `R` in the primary method. Predictive state covariance is considered separately only if mean-state learning is stable and the mentor confirms it as a core deliverable.
- The paper's ESKF-final-output architecture does not establish that a PINN can directly produce the final deployed state without a deployment ESKF.
- The paper does not establish strict `SO(2)` equivariance, no-high-precision-supervision training, or this project's 20-/30-second outage success criteria.

**Permitted project wording:** PINK-GINS is a first-party antecedent for PINN-constrained tightly coupled GNSS/INS and explicit strapdown residuals, but its ground-truth supervision and jointly learned bias/noise design are materially different from the candidate method.

## AutoW

**Work:** Penghui Xu and Li-Ta Hsu, "AutoW: Self-Supervision Learning for Weighting Estimation in GNSS Positioning," ION GNSS+ 2024, pp. 2630-2644, DOI 10.33012/2024.19896.

**Primary sources checked on 2026-08-22:**

- ION official peer-reviewed publication record: <https://www.ion.org/publications/abstract.cfm?articleID=19896>
- Author's 2025 Hong Kong Polytechnic University Ph.D. thesis record: <https://theses.lib.polyu.edu.hk/handle/200/13752>

**Verified method facts:**

- AutoW learns pseudorange measurement weights through a differentiable factor-graph optimization pipeline without manually supplied position labels for the declared AutoW experiment.
- Its self-supervised loss is built from two priors explicitly associated with static experiments: solution clustering and zero velocity.
- The ION evaluation reported smartphone data from an OPPO Find X6; the institutional thesis additionally describes AutoW evaluation with u-blox F9P and OPPO data.

**Admissible influence on this project:**

- Demonstrates that a deployable physical condition can provide a useful training signal without trajectory labels when that condition is independently valid.
- Provides a comparison point for differentiable estimation and learned GNSS weighting, but only under its declared static-condition assumptions.

**Not supported by this source:**

- Static clustering and zero velocity are not valid for ordinary moving-vehicle intervals. Applying them to all driving data would force real motion toward a false static solution.
- It does not provide a general dynamic-vehicle GNSS/INS physics loss, a 20--30 second moving outage solution, an ESKF-pseudo-label state estimator, or a strict `SO(2)` architecture.
- It does not justify allowing a network to determine its own static label and benefit from the resulting loss, nor does it make a learned-weight solution the canonical fixed SPP baseline.

**Permitted project wording:** AutoW is a first-party example of condition-specific self-supervision for GNSS weighting. Its static priors are not transferred into the primary moving-vehicle loss; ZUPT remains deferred until an independent deployable static detector is verified.

## TLIO

**Work:** Wenxin Liu, David Caruso, Eddy Ilg, Jing Dong, Anastasios I. Mourikis, Kostas Daniilidis, Vijay Kumar, and Jakob Engel, "TLIO: Tight Learned Inertial Odometry," IEEE Robotics and Automation Letters 5(4), 2020, pp. 5653-5660, DOI 10.1109/LRA.2020.3007421.

**Primary sources checked on 2026-08-22:**

- Author paper record/preprint: <https://arxiv.org/abs/2007.01867>
- Author official supplementary repository: <https://github.com/CathIAS/TLIO>

**Verified method facts:**

- TLIO is an IMU-only deployment method, not a GNSS method. A neural network regresses a three-dimensional relative displacement measurement and its uncertainty from an IMU segment.
- The learned relative measurement is fused in a stochastic-cloning EKF that estimates pose, velocity, and IMU biases. Here "tight" describes integration of the learned inertial measurement with the filter, not per-satellite GNSS/INS tight coupling.
- The official repository's processed training array contains VIO ground-truth quantities. It also warns that standalone network trajectory testing uses ground-truth orientations and is a debugging path rather than a benchmark.

**Admissible influence on this project:**

- Supports the concept of a learned relative inertial prior with predicted uncertainty and an explicit state estimator.
- Supports auditing the exact network/filter information interface and distinguishing a learned relative measurement from a complete absolute navigation state.
- Provides an architectural ancestor for EqNIO's filter-based application and a supervised comparison point for displacement-prior methods.

**Not supported by this source:**

- It is not a no-high-precision-trajectory-supervision precedent: the released training representation includes processed VIO truth.
- It does not fuse GNSS, address urban GNSS measurement degradation, provide ESKF weak-pseudo-label distillation, or define a PINN inertial residual.
- A finite IMU window does not by itself supply the absolute position/velocity integration constants required by this project's final state; TLIO's filter state and initialization remain essential.

**Permitted project wording:** TLIO is a supervised, filter-fused neural displacement-prior antecedent. It informs the relative-prior interface but not the candidate training objective or GNSS-outage claim.

## RoNIN

**Work:** Sachini Herath, Hang Yan, and Yasutaka Furukawa, "RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods," IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 3146-3152.

**Primary sources checked on 2026-08-22:**

- Author manuscript: <https://arxiv.org/abs/1905.12853>
- Author/institution project page: <https://ronin.cs.sfu.ca/>
- Author official repository: <https://github.com/Sachini/ronin>

**Verified method facts:**

- RoNIN is an inertial-navigation method whose position networks take IMU history and regress a two-dimensional velocity or one-second positional difference; the predicted horizontal motion is integrated into a trajectory.
- Training is supervised by high-quality trajectory data. The LSTM/TCN latent-velocity loss matches the integrated outputs to ground-truth position differences, while the ResNet strided loss directly compares its output with a ground-truth position difference. The paper also reports ground-truth instantaneous velocity supervision in ablations.
- Its heading-agnostic coordinate frame aligns the vertical axis with gravity and applies random horizontal rotations during training. At inference it depends on a device-orientation estimate supplied by the phone system.
- The RoNIN benchmark itself contains ground-truth 3D trajectories, and parts of the training-time orientation preparation can select or use ground-truth orientation when the estimated orientation is judged too poor.

**Admissible influence on this project:**

- Supports representing IMU history and horizontal motion in a gravity-aligned frame, and it is an important supervised antecedent for IMU-to-velocity temporal networks.
- Motivates explicitly testing whether orientation normalization, augmentation, and structural equivariance make distinct contributions.

**Not supported by this source:**

- It is not a no-high-precision-truth training method: ground-truth position differences, velocities, headings, and in some cases orientations affect training or data preparation.
- Random horizontal rotation and coordinate normalization provide empirical rotational robustness but do not prove that every layer commutes with the continuous `SO(2)` action.
- It does not fuse GNSS, model GNSS outages, produce a full GNSS/INS position-velocity-attitude state, or supply this project's ESKF-pseudo-label plus inertial-physics objective.
- Its human-held-smartphone setting and two-dimensional integrated output do not directly establish robustness for vehicle navigation in urban GNSS degradation.

**Permitted project wording:** RoNIN is a supervised IMU-to-horizontal-velocity/relative-motion antecedent using gravity-aligned coordinate normalization and rotation augmentation. EqNIO strengthens the geometric treatment structurally; neither RoNIN nor EqNIO supplies this project's no-high-precision-truth training objective.

## RINS-W

**Work:** Martin Brossard, Axel Barrau, and Silvere Bonnabel, "RINS-W: Robust Inertial Navigation System on Wheels," IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 2068-2075, DOI 10.1109/IROS40897.2019.8968593.

**Primary sources checked on 2026-08-22:**

- Author manuscript: <https://arxiv.org/abs/1903.02210>
- Author official repository: <https://github.com/mbrossar/RINS-W>

**Verified method facts:**

- RINS-W performs IMU-only dead reckoning for wheeled vehicles. Its neural component detects four motion profiles: zero linear velocity, zero angular velocity, approximately zero lateral velocity, and approximately zero vertical velocity.
- The detector is trained as four binary classifiers using binary cross-entropy. Its labels are produced by differentiating and smoothing ground-truth poses and thresholding the resulting ground-truth linear/angular velocities.
- The neural detector does not output the final navigation state. An invariant extended Kalman filter consumes detected motion profiles as pseudo-measurements and outputs pose, velocity, IMU biases, and covariance.
- The original paper assumes the vehicle body frame is aligned with the IMU frame, ignores Earth rotation and Coriolis effects, and evaluates data from the same car and IMU family; it explicitly leaves generalization to other platforms and sensors unresolved.
- The later official repository differs from the paper: its neural network detects only zero velocity, while no-lateral-slip and vertical-velocity assumptions remain filter pseudo-measurements.

**Admissible influence on this project:**

- Supports the principle that a motion constraint must first be independently detected and should enter an explicit estimator as a conditional pseudo-measurement, rather than being imposed universally.
- Supports distinguishing learned condition detection from end-to-end state prediction and separately measuring detector errors and navigation errors.

**Not supported by this source:**

- It is not a no-high-precision-truth training precedent because ground-truth poses determine every detector label.
- It does not fuse GNSS, directly output a final PINN trajectory, establish strict `SO(2)` equivariance, or provide a physics residual that trains a full state network without condition labels.
- Its body/IMU co-alignment, flat-Earth model, and universal wheeled-vehicle assumptions cannot replace this project's explicit installation extrinsics, Earth terms, or validity gates.
- It does not justify adding ZUPT or non-holonomic constraints to the primary method without an independently available deployable detector and verified reference-point/extrinsic information.

**Permitted project wording:** RINS-W is a supervised motion-profile-detector plus IEKF antecedent. It motivates conditional constraint handling, but ZUPT and NHC remain outside this project's primary method under the mentor's current instruction.

## AI-IMU Dead-Reckoning

**Work:** Martin Brossard, Axel Barrau, and Silvere Bonnabel, "AI-IMU Dead-Reckoning," IEEE Transactions on Intelligent Vehicles 5(4), 2020, pp. 585-595, DOI 10.1109/TIV.2020.2980758.

**Primary sources checked on 2026-08-22:**

- Author manuscript: <https://arxiv.org/abs/1904.06064>
- Author official repository: <https://github.com/mbrossar/ai-imu-dr>

**Verified method facts:**

- AI-IMU Dead-Reckoning is an IMU-only wheeled-vehicle method. An IEKF integrates IMU data and treats approximately zero lateral and vertical vehicle-frame velocities as pseudo-measurements.
- A causal CNN adapter maps a short raw-IMU window to a bounded diagonal covariance for those two pseudo-measurements. The adapter does not receive the filter state.
- The IEKF, not the network, produces position, velocity, orientation, IMU-bias, installation-state, and covariance estimates.
- Training backpropagates through the filter and minimizes relative translation error against accurate reference poses. The paper explicitly envisages precise GNSS or LiDAR poses as the training truth.
- The optimized variables include the adapter parameters, initial covariance `P0`, and fixed process-noise entries `Q`. The paper explicitly warns that adapter outputs are chosen to improve localization accuracy and may differ substantially from the actual statistical pseudo-measurement covariance.

**Admissible influence on this project:**

- Supports causal, state-independent IMU conditioning of bounded uncertainty parameters and differentiable training through a state estimator.
- Demonstrates why a learned covariance-like control can be useful operationally while still requiring a separate statistical-calibration audit.
- Provides a relevant controlled-outage comparison point for wheeled-vehicle IMU dead reckoning, subject to matching information, initialization, data splits, and evaluation rules.

**Rejected or modified design choices:**

- Its trajectory-truth loss is prohibited for this project's training, tuning, and checkpoint selection; an ESKF pseudo-label is not equivalent to its accurate-reference supervision.
- Joint optimization of the covariance adapter, `P0`, and `Q` is not copied because they can compensate for one another and obscure physical meaning.
- A quantity optimized solely for trajectory error is not reported as calibrated predictive covariance. If a project covariance head is later approved, it must be developed separately from the mean model and evaluated for blind coverage, sharpness, and proper scoring.
- Universal zero-lateral/vertical pseudo-measurements are not admitted to the primary method under the mentor's current instruction; any later use requires independent validity gates, vehicle-reference-point and installation-extrinsic audits.
- The paper's flat-Earth mechanics and ground-truth initialization do not replace this project's fixed-NED0 Earth terms or common deployable initialization policy.

**Permitted project wording:** AI-IMU Dead-Reckoning is a supervised differentiable-filter antecedent for IMU-conditioned pseudo-measurement covariance adaptation. It does not support no-high-precision-truth training or treating performance-tuned covariance controls as calibrated state uncertainty.

## RIO (rotation-equivariance supervised inertial odometry)

**Work:** Xiya Cao, Caifa Zhou, Dandan Zeng, and Yongliang Wang, "RIO: Rotation-Equivariance Supervised Learning of Robust Inertial Odometry," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 6614-6623.

**Primary sources checked on 2026-08-22:**

- CVF official proceedings record and paper: <https://openaccess.thecvf.com/content/CVPR2022/html/Cao_RIO_Rotation-Equivariance_Supervised_Learning_of_Robust_Inertial_Odometry_CVPR_2022_paper.html>
- Author manuscript: <https://arxiv.org/abs/2111.11676>
- EqNIO first-party comparison appendix: <https://proceedings.iclr.cc/paper_files/paper/2025/file/6554c4151c2ccbd36da2d55c015b2039-Paper-Conference.pdf>

**Verified method facts:**

- RIO maps a gravity-aligned IMU window to a velocity estimate. It creates yaw-rotated copies of an input and penalizes disagreement between the rotated original prediction and the prediction made from the rotated input, using negative cosine similarity.
- The authors explicitly state that this auxiliary relation alone cannot determine a consistent real velocity direction and magnitude. Ordinary training therefore jointly includes an MSE stride-velocity loss against ground-truth trajectory-derived velocity.
- The auxiliary loss is disabled at low predicted velocity because direction becomes ambiguous. At inference, RIO performs test-time training on batches of recent samples using four fixed rotations and may restore the pretrained model based on an ensemble-derived uncertainty heuristic.
- Its uncertainty indicator is the variance across separately initialized velocity networks. It is used to trigger or suppress online parameter updates; it is not a full navigation-state covariance with a demonstrated probabilistic calibration guarantee.
- EqNIO's first-party appendix classifies RIO as approximate/soft equivariance, in contrast to EqNIO's structural commuting layers.

**Admissible influence on this project:**

- Provides direct evidence that rotation-consistency can be a useful auxiliary signal and motivates the pre-registered comparison among ordinary networks, rotation augmentation/auxiliary loss, and strict structural `SO(2)` equivariance.
- Demonstrates that unlabeled rotation consistency may support adaptation, while also making clear that it does not identify the physical velocity scale or absolute state by itself.

**Rejected or modified design choices:**

- RIO is not cited as training without trajectory truth: its principal training regime retains ground-truth velocity MSE.
- A consistency loss evaluated on finitely rotated examples does not prove continuous `SO(2)` equivariance and cannot replace architecture-level group-commutation tests.
- The primary method keeps inference weights frozen. Test-time parameter updates, uncertainty-triggered self-gating, and model resets would change the deployed method online and require a separate pre-registered extension with causal, safety, and reproducibility audits.
- Ensemble spread is not reported as calibrated full-state covariance. Any later approved project covariance output would require blind coverage, sharpness, and proper-scoring checks.

**Permitted project wording:** RIO is a supervised IMU-to-velocity antecedent augmented by rotation-consistency self-supervision and test-time adaptation. It is the soft-equivariance comparison point, not evidence that rotation consistency alone trains a complete navigation state or guarantees strict `SO(2)` equivariance.

## IDOL (Inertial Deep Orientation-Estimation and Localization)

**Work:** Scott Sun, Dennis Melamed, and Kris Kitani, "IDOL: Inertial Deep Orientation-Estimation and Localization," Proceedings of the AAAI Conference on Artificial Intelligence 35(7), 2021, pp. 6128-6137, DOI 10.1609/aaai.v35i7.16763.

**Primary sources checked on 2026-08-22:**

- AAAI official proceedings record and paper: <https://ojs.aaai.org/index.php/AAAI/article/view/16763>
- Author manuscript: <https://arxiv.org/abs/2102.04024>
- Author official repository: <https://github.com/KlabCMU/IDOL>

**Verified method facts:**

- IDOL is a pedestrian/smartphone pipeline with separate orientation and position stages. Its orientation network consumes accelerometer, gyroscope, and magnetometer readings, predicts an absolute orientation and orientation-error covariance, and fuses that prediction with gyroscope propagation in an EKF.
- The orientation mean and covariance are trained by a Gaussian NLL against ground-truth device orientation. The position network predicts two-dimensional window-relative displacement and is trained by MSE against ground-truth displacement.
- Its data truth comes from a rigidly attached LiDAR/video/Xsens SLAM rig validated in a Vicon studio. The authors explicitly describe the overall approach as supervised.
- The position module uses a bidirectional LSTM over a window, then accumulates relative displacements. This architecture is not a strictly causal streaming operator at each sample inside the window.
- The paper reports failure under magnetic-field changes in unseen buildings and position-regression degradation across building geometries.

**Admissible influence on this project:**

- Supports treating orientation quality as a first-order source of position error and auditing attitude, position, and covariance outputs separately.
- Provides an antecedent for manifold-compatible orientation innovations and an NLL-trained attitude uncertainty head when valid truth is available.

**Not supported by this source:**

- It is not a no-high-precision-truth training precedent; both its orientation and displacement heads use accurate pose truth.
- It uses a magnetometer outside this project's frozen six-axis IMU plus GNSS information contract, and its absolute-heading mechanism is vulnerable to environment-specific magnetic disturbances.
- Its bidirectional position network does not satisfy the candidate method's strict causal tail-window rule.
- It does not fuse GNSS, impose independent strapdown physics residuals, guarantee strict `SO(2)` equivariance, or output the candidate method's full position-velocity-attitude state.

**Permitted project wording:** IDOL is a supervised, magnetometer-aided orientation-EKF plus bidirectional displacement-network antecedent. It demonstrates the importance of attitude quality, but not the feasibility of this project's deployable information policy, causal architecture, or no-high-precision-truth objective.

## MoRPI-PINN

**Work:** Arup Kumar Sahoo and Itzik Klein, "MoRPI-PINN: a physics-informed framework for mobile robot pure inertial navigation," Scientific Reports 16, Article 19827, 2026, DOI 10.1038/s41598-026-50630-y.

**Primary source checked on 2026-08-22:**

- Publisher version of record: <https://www.nature.com/articles/s41598-026-50630-y>

**Verified method facts:**

- MoRPI-PINN is a pure-inertial, planar mobile-robot state network. It takes time, horizontal specific-force components, and vertical angular rate as inputs and predicts two-dimensional position, two-dimensional velocity, and yaw.
- Its physics loss genuinely depends on learned state outputs. It penalizes `d(p_hat)/dt - v_hat`, `d(v_hat)/dt - (C(psi_hat) f + g)`, and `d(psi_hat)/dt - omega_z`, using automatic differentiation.
- Its total objective is not truth-free. A data term applies MSE between predicted and ground-truth position/velocity, and an initial-condition term uses known starting position. The data were collected with a Javad SIGMA-3N RTK-GNSS receiver whose positions serve as ground truth.
- The network is a fully connected PINN trained on overlapping windows. Time is a direct input; axes are normalized independently; validation behavior controls learning-rate scheduling, early stopping, and empirical hyperparameter selection.
- The application deliberately exploits prescribed snake-like/periodic robot motion. Its mechanics assume planar motion, neglect roll, pitch, vertical dynamics, Earth rotation, and transport rate.

**Admissible influence on this project:**

- Provides a recent first-party precedent for an independent PINN residual whose value and gradient depend directly on predicted navigation states and measured IMU quantities.
- Supports including separate position-kinematic, velocity-dynamic, and attitude-rate residual blocks and ablating the physics term against data-only training.

**Rejected or modified design choices:**

- RTK-derived position/velocity may not affect this project's training, tuning, early stopping, checkpoint selection, or normalizer fitting. MoRPI-PINN therefore does not meet the candidate information policy.
- Direct time input can encode trajectory phase or periodic-route shortcuts and remains prohibited in this project's network information contract; only causal time intervals and masks are retained.
- Independent horizontal-axis normalization and ordinary fully connected/ReLU layers do not enforce continuous `SO(2)` equivariance.
- Planar snake-motion assumptions cannot replace the candidate's three-dimensional strapdown state, Earth terms, installation extrinsics, arbitrary ordinary vehicle motion, or GNSS-conditioned posterior correction.
- MoRPI-PINN does not provide GNSS/INS fusion during deployment, ESKF pseudo-label weak supervision, full attitude, or the candidate outage/available-GNSS co-primary evaluation.

**Permitted project wording:** MoRPI-PINN is the closest verified recent antecedent for direct state-output PINN residuals in pure planar inertial navigation. The candidate contribution must not claim first use of PINN state dynamics for inertial navigation; its distinguishers are the no-high-precision-development policy, GNSS-plus-IMU causal full-state architecture, ESKF weak labels, strict `SO(2)` structure, and pre-registered 20-/30-second outage evaluation.

## Seamless Indoor and Outdoor Navigation Using IMU-GNSS Sensor Data Fusion

**Work:** Bismark Kweku Asiedu Asante and Hiroki Imamura, "Seamless Indoor and Outdoor Navigation Using IMU-GNSS Sensor Data Fusion," Sensors 26(7), 2215, 2026, DOI 10.3390/s26072215.

**Primary sources checked on 2026-08-22:**

- Publisher full-text record: <https://www.mdpi.com/1424-8220/26/7/2215>
- PubMed record: <https://pubmed.ncbi.nlm.nih.gov/41978000/>

**Verified method facts:**

- The framework contains a PINN correction module and an EKF. The PINN consumes synchronized IMU/GNSS information together with the previous state/control representation and predicts corrected next-state or pseudo-measurement quantities; the EKF performs the final probabilistic state estimation and outputs position, velocity, and orientation.
- GNSS enters as latitude, longitude, and altitude converted to an absolute three-dimensional position. The disclosed observation interface does not use per-satellite pseudorange, Doppler, carrier phase, clock, or satellite geometry.
- The PINN is a four-hidden-layer fully connected tanh MLP. Training uses KITTI IMU/GNSS data with chronological 75/15/15 partitions and validation-dependent convergence/hyperparameter decisions.
- The declared loss includes reference-position, reference-orientation, and reference-acceleration MSE terms. A soft gate changes the balance between GNSS-dependent supervision and the declared physics terms according to GNSS availability/quality.
- Evaluation also uses a custom indoor/outdoor pedestrian dataset whose outdoor reference is GNSS and whose indoor reference includes predefined routes.

**Admissible influence on this project:**

- Provides a recent GNSS-plus-IMU PINN/EKF comparison point and motivates explicit testing of GNSS-available, GNSS-denied, and reacquisition-transition regimes.
- Supports keeping the learned correction role and final estimator role explicit rather than calling the whole pipeline a single undifferentiated PINN.

**Terminology and evidence limitations:**

- Despite the paper's use of "tightly coupled" for PINN/EKF integration, its disclosed GNSS interface is PVT position. Under standard GNSS/INS terminology this is loose coupling; it is not evidence for a per-satellite tightly coupled method.
- Reference position/orientation/acceleration MSE terms are supervised signals. Calling their sum a physics loss does not make them independent governing-equation residuals, and the work does not meet this project's no-high-precision-development policy.
- The EKF, not the PINN, is the final state estimator, so the architecture does not anticipate the mentor-directed deployed PINN-final-state design.
- The published table reports mean position error values larger than the corresponding RMSE values even though RMSE is defined from the same nonnegative Euclidean position errors. Since RMS must be at least the arithmetic mean on the same samples, these reported pairs are internally inconsistent as written. They are not used as performance evidence here.
- Chronological point partitioning alone does not establish route/session-level isolation when adjacent motion samples or windows may share trajectory context.
- The work does not provide strict `SO(2)` equivariance, a sealed high-precision evaluation policy, or this project's pre-registered paired 20-/30-second outage protocol.

**Permitted project wording:** This Sensors paper is a recent supervised loose-GNSS-position PINN-plus-EKF antecedent for indoor/outdoor transition handling. It is not a per-satellite tightly coupled method, a no-high-precision-development method, or a PINN-final-state precedent; its published numerical comparisons require caution because the stated mean/RMSE pairs are internally inconsistent.

## SINS/GNSS Left Equivariant Error Model in Inertial Frame

**Work:** Yarong Luo, Chi Guo, Wentao Lu, and Yulong Huang, "SINS/GNSS Integrated Navigation Based on Left Equivariant Error Model in Inertial Frame," IEEE Transactions on Intelligent Transportation Systems, Early Access, 2026, DOI 10.1109/TITS.2026.3701149.

**First-party sources checked on 2026-08-22:**

- IEEE Xplore accepted-version full paper, 17 pages, carrying DOI 10.1109/TITS.2026.3701149 and the IEEE publication footer
- IEEE publisher record and abstract: <https://doi.org/10.1109/TITS.2026.3701149>
- Wuhan University corresponding-author publication record: <https://jszy.whu.edu.cn/guochi/en/zhym/278585/list/index.htm>
- Author laboratory technical note: <https://zhiyuteam.com/html/web/yanjiuchengguo/qikanlunwen/2064160895785439234.html>

**Verified method facts:**

- This is a classical equivariant filter, not a neural network. Its biased-INS state manifold is `SE_2(3) x se_2(3)`, containing attitude, inertial-frame velocity and position, and body-frame gyroscope/accelerometer biases. A two-frame symmetry group `G_TFG` acts on the state and input spaces, and the estimator uses a left-equivariant global error with an 18-dimensional local coordinate.
- Gyroscope and accelerometer biases are modeled as random walks. The inertial-frame formulation retains Earth rotation in the gravitation-vector transformation, while the gravitation gradient is explicitly ignored during uncertainty propagation. The construction introduces virtual nonphysical bias inputs to define its group action and lift.
- The paper explicitly states that the `G_TFG` group structure still introduces linearization error. Its distinction from the cited imperfect invariant EKF is that the derivation does not assume a small attitude misalignment; the resulting linearized error dynamics depend on the estimated biases but not on the navigation state.
- Its output-equivariance result concerns the filter configuration output `h(xi)=Tc` under the declared `G_TFG` action `rho(X,y)=Cy`. This is a property of the state/output geometry used in filter design, not a learned-map commutation result.
- The loose model uses absolute GNSS antenna position transformed from ECEF to the inertial frame and explicitly includes the antenna-to-IMU lever arm. The derived position measurement matrix is independent of the estimated position.
- The paper's tight model is specifically short-baseline, double-difference SINS/RTK. It uses double-difference pseudorange and carrier phase, a known reference-station position, broadcast/product satellite positions, and single-epoch ambiguity resolution; ambiguity parameters are treated as independent across epochs. It is not the same observation contract as a standalone-receiver pseudorange/Doppler tightly coupled method.
- The reset version `I-TFG` parallel-transports the updated covariance using a Cartan-Schouten (+) connection; `Im-TFG` omits this reset. The declared classical comparisons are ECEF EKF, ECEF left imperfect invariant EKF, inertial-frame left imperfect invariant EKF, `Im-TFG`, and `I-TFG`.

**Verified experiment facts and limits:**

- The simulation includes small- and large-misalignment cases, signal blockages, multipath, gross errors, and a complete 20-second GNSS denial from 280 to 300 seconds. It uses 100 Monte Carlo runs for attitude and position NEES summaries. It does not test a learned estimator or the candidate's co-primary 20-/30-second paired outage protocol.
- The field evaluation uses UrbanNav-HK-Medium and UrbanNav-HK-Harsh, with 1 Hz u-blox ZED-F9P GNSS and 400 Hz Xsens MTi-10 IMU. Inertial Explorer post-processed solutions serve as reference.
- The field-test gyroscope and accelerometer constant biases are derived from post-processed high-precision navigation solutions. Therefore the experiment does not satisfy this project's rule that high-precision information may not determine development parameters, initialization, training, tuning, or model selection.
- The paper's reported advantage is strongest in the deliberately large initial-misalignment case and is attributed to the trajectory-independent linearized error dynamics, rotated bias-error definition, and covariance reset. These results do not establish neural `SO(2)` equivariance, PINN physics learning, truth-free model development, or superiority of a PINN-final estimator.

**Boundary for this project:**

- Here, `equivariant error/filter` refers to symmetry-aware state/error geometry and filter linearization. The candidate's strict neural `SO(2)` equivariance instead requires its learned input-output map to commute with one fixed horizontal rotation acting on every typed directional quantity in the causal window.
- Sharing group-action terminology does not make the two properties interchangeable. A left-equivariant filter does not prove that an ordinary neural network is `SO(2)`-equivariant, and a neural `SO(2)` commutation test does not establish the full-state Lie-group filter's consistency or large-misalignment properties.
- This paper may motivate a future classical comparator or sensitivity analysis. It does not authorize changing the frozen conventional teacher ESKF, its pseudo-label definition, or the primary baselines during Phase 1.
- Its short-baseline double-difference SINS/RTK equations cannot be cited as if they implemented the candidate's different per-satellite pseudorange/Doppler extension.

**Permitted project wording:** Recent SINS/GNSS work also uses equivariance at the filter-error geometry level and has demonstrated classical filter benefits under large initial misalignment. That use is related through group actions but is distinct from, and does not substitute for, the candidate network's strict causal horizontal-rotation `SO(2)` equivariance. Its tight component is short-baseline double-difference SINS/RTK rather than the candidate's standalone-receiver pseudorange/Doppler extension, and its field-test parameterization uses post-processed high-precision information.
