from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from pv_surrogate_eurocast.constants import Paths, SystemData


def create_pvgis_parameters_vs_error_plot(results: Path, target_path: Path, parameters: list[str], errors: list[str]):
    results = pd.read_parquet(results)
    print(results.iloc[0]['model'])
    print(results.columns)

    target_path = target_path / results.iloc[0]['model']
    target_path.mkdir(parents=True, exist_ok=True)
    target_path = target_path
    print(results[errors].mean())

    # Create a single figure with len(errors) subplots
    fig, axes = plt.subplots(1, len(errors), figsize=(15, 5))

    for error in errors:
        for i, parameter in enumerate(parameters):
            scaler = MinMaxScaler()
            e = scaler.fit_transform(results['MAPE'].values.reshape(-1, 1))
            r = pd.concat([results, pd.Series(e.flatten(), name=f'N_{error}')], axis=1)
            r['marker'] = r[f'N_{error}'] > r[f'N_{error}'].mean()
            sns.histplot(data=r, x=parameter, hue='marker', fill=True, palette='muted', ax=axes[i])
            axes[i].set_xlabel(parameter)
            axes[i].set_ylabel('Count')
        plt.suptitle(f'Distribution of Above Average {error} ({r[error].mean()})')
        plt.savefig(target_path / f'{error}.pdf')


def main():
    parameters = ["kwP", "orientation", "tilt"]
    results_path = Paths.figure_dir / "parameters" / "symreg" / 'fixed'
    results_path.mkdir(parents=True, exist_ok=True)
    # create_pvgis_parameters_vs_error_plot(
        # Paths.fixed_test_results_symreg_parquet, results_path, parameters, ["MAPE", "MAE", "R2"]
    # )
    results_path = Paths.figure_dir / "location" / "symreg" / 'location'
    results_path.mkdir(parents=True, exist_ok=True)
    create_pvgis_parameters_vs_error_plot(
        Paths.pvgis_outward_data_dir / 'evaluation.parquet', results_path, ['lon', 'lat'], ["MAPE", "MAE", "R2"]
    )


if __name__ == "__main__":
    main()
