import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from sklearn.model_selection import train_test_split


class DistanceBasedKMeansClassifier(BaseEstimator, ClassifierMixin):
    """KMeans proxy with distance-based soft probabilities."""

    def __init__(self, centers, temperature=1.0):
        self.centers = centers
        self.temperature = temperature
        self.classes_ = np.arange(len(centers))
        self.n_features_in_ = centers.shape[1]

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        dists = cdist(X, self.centers, metric='euclidean')
        return np.argmin(dists, axis=1)

    def predict_proba(self, X):
        dists = cdist(X, self.centers, metric='euclidean')
        squared_dists = dists ** 2
        neg_dists = -squared_dists / self.temperature
        neg_dists_stable = neg_dists - np.max(neg_dists, axis=1, keepdims=True)
        exp_dists = np.exp(neg_dists_stable)
        probs = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
        return probs


def evaluate_membership_inference_attack(
        X_train,
        X_test,
        centers,
        attack_train_ratio=0.5,
        attack_model_type='rf',
        temperature=1.0,
        random_state=42,
        **attack_kwargs
):
    """
    Evaluate MIA with BALANCED attack training set.

    Key fix: Ensures equal numbers of members and non-members for attack training.
    """

    # Step 1: Determine balanced sizes
    n_members_total = len(X_train)
    n_non_members_total = len(X_test)

    # Use the smaller class to determine max balanced size
    max_balanced_size = min(n_members_total, n_non_members_total)

    # Calculate how many samples per class for attack training
    n_per_class_train = int(max_balanced_size * attack_train_ratio)
    n_per_class_eval = max_balanced_size - n_per_class_train

    print(f"\n{'=' * 70}")
    print(f"Membership Inference Attack Configuration (BALANCED)")
    print(f"{'=' * 70}")
    print(f"  Available: {n_members_total} members, {n_non_members_total} non-members")
    print(f"  Max balanced size: {max_balanced_size} per class")
    print(f"  Attack train ratio: {attack_train_ratio}")
    print(f"  Random state: {random_state}")
    print(f"  Attack model type: {attack_model_type}")
    print(f"  Temperature: {temperature}")

    # Step 2: Shuffle and sample balanced subsets
    rng = np.random.RandomState(random_state)

    # Shuffle members
    member_indices = rng.permutation(n_members_total)
    selected_members = X_train[member_indices[:max_balanced_size]]

    # Shuffle non-members
    non_member_indices = rng.permutation(n_non_members_total)
    selected_non_members = X_test[non_member_indices[:max_balanced_size]]

    # Step 3: Split into attack train/eval (balanced)
    X_train_members_attack = selected_members[:n_per_class_train]
    X_train_members_eval = selected_members[n_per_class_train:]

    X_test_non_members_attack = selected_non_members[:n_per_class_train]
    X_test_non_members_eval = selected_non_members[n_per_class_train:]

    print(f"\n  BALANCED Attack Training:")
    print(f"    {len(X_train_members_attack)} members + {len(X_test_non_members_attack)} non-members")
    print(f"    = {len(X_train_members_attack) + len(X_test_non_members_attack)} total (50-50 split)")

    print(f"\n  BALANCED Attack Evaluation:")
    print(f"    {len(X_train_members_eval)} members + {len(X_test_non_members_eval)} non-members")
    print(f"    = {len(X_train_members_eval) + len(X_test_non_members_eval)} total (50-50 split)")

    # Step 4: Create proxy classifier
    proxy = DistanceBasedKMeansClassifier(centers, temperature=temperature)
    proxy.fit(centers)
    art_classifier = ScikitlearnClassifier(model=proxy)

    # Step 5: Distance statistics (on evaluation set)
    train_dists = cdist(X_train_members_eval, centers, metric='euclidean')
    test_dists = cdist(X_test_non_members_eval, centers, metric='euclidean')

    train_min_dists = np.min(train_dists, axis=1)
    test_min_dists = np.min(test_dists, axis=1)

    print(f"\n  Distance Statistics (Evaluation Set):")
    print(f"    Members     - Mean: {np.mean(train_min_dists):.4f}, Std: {np.std(train_min_dists):.4f}")
    print(f"    Non-members - Mean: {np.mean(test_min_dists):.4f}, Std: {np.std(test_min_dists):.4f}")
    print(f"    Difference:  {np.mean(test_min_dists) - np.mean(train_min_dists):.4f}")

    # Calculate Cohen's d
    mean_diff = np.mean(test_min_dists) - np.mean(train_min_dists)
    pooled_std = np.sqrt((np.std(train_min_dists) ** 2 + np.std(test_min_dists) ** 2) / 2)
    cohens_d = mean_diff / (pooled_std + 1e-10)
    print(f"    Cohen's d:   {cohens_d:.4f}", end="")

    if abs(cohens_d) < 0.2:
        print(" (negligible effect - excellent privacy)")
    elif abs(cohens_d) < 0.5:
        print(" (small effect)")
    elif abs(cohens_d) < 0.8:
        print(" (medium effect)")
    else:
        print(" (large effect - privacy risk)")

    # Step 6: Create and train attack
    attack = MembershipInferenceBlackBox(
        estimator=art_classifier,
        attack_model_type=attack_model_type,
        **attack_kwargs
    )

    print(f"\n  Training attack model...")
    attack.fit(
        x=X_train_members_attack,
        y=None,
        test_x=X_test_non_members_attack,
        test_y=None
    )

    # Step 7: Evaluate attack
    print(f"  Running inference...")
    inferred_train = attack.infer(x=X_train_members_eval, y=None)
    inferred_test = attack.infer(x=X_test_non_members_eval, y=None)

    # Step 8: Calculate metrics
    train_acc = np.mean(inferred_train)  # TPR
    test_acc = 1 - np.mean(inferred_test)  # TNR
    balanced_acc = (train_acc + test_acc) / 2

    total_predicted_members = np.sum(inferred_train) + np.sum(inferred_test)
    total_samples = len(inferred_train) + len(inferred_test)

    print(f"\n{'=' * 70}")
    print(f"Membership Inference Attack Results")
    print(f"{'=' * 70}")
    print(f"  MIA Train (Member) Accuracy (TPR):     {train_acc:.4f}")
    print(f"  MIA Test (Non-Member) Accuracy (TNR):  {test_acc:.4f}")
    print(f"  Balanced MIA Accuracy:                 {balanced_acc:.4f}")
    print(f"  Total predicted as members:            {total_predicted_members:.0f}/{total_samples}")
    print(
        f"  Prediction bias:                       {(total_predicted_members / total_samples - 0.5) * 100:+.1f}% towards members")
    print(f"{'=' * 70}")

    # Step 9: Interpretation
    if balanced_acc < 0.55:
        privacy_level = "EXCELLENT ✓✓"
        recommendation = "DP noise is very effective at hiding membership"
        symbol = "✓✓"
    elif balanced_acc < 0.60:
        privacy_level = "GOOD ✓"
        recommendation = "DP noise provides reasonable protection"
        symbol = "✓"
    elif balanced_acc < 0.70:
        privacy_level = "MODERATE ⚠"
        recommendation = "Consider increasing dp_sigma"
        symbol = "⚠"
    else:
        privacy_level = "WEAK ✗"
        recommendation = "Increase dp_sigma significantly"
        symbol = "✗"

    print(f"\n  {symbol} Privacy Level: {privacy_level}")
    print(f"     Recommendation: {recommendation}")

    # Additional insight based on Cohen's d
    if abs(cohens_d) < 0.2 and balanced_acc < 0.55:
        print(f"\n  ✓ Confirmed: Distance distributions are nearly identical")
        print(f"     This indicates effective privacy protection.")
    elif abs(cohens_d) > 0.5 and balanced_acc > 0.65:
        print(f"\n  ✗ Warning: Significant distance gap persists")
        print(f"     Members are detectably closer to centers.")
        print(f"     Recommendation: Increase DP noise or normalize data scale")

    print(f"{'=' * 70}\n")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'balanced_acc': balanced_acc,
        'cohens_d': cohens_d,
        'privacy_level': privacy_level,
        'train_distances': train_min_dists,
        'test_distances': test_min_dists,
        'prediction_counts': {
            'members_predicted': int(total_predicted_members),
            'total_samples': int(total_samples)
        }
    }


# Usage
if __name__ == "__main__":
    results = evaluate_membership_inference_attack_balanced(
        X_train=X_train,
        X_test=X_test,
        centers=centers_orig,
        attack_train_ratio=0.5,
        attack_model_type='rf',
        temperature=1.0,
        random_state=42
    )