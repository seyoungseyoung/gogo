from argparse import ArgumentParser
from cluster import run_HDBSCAN
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Cluster embeddings using a pre-generated reference file."
    )
    parser.add_argument('-r', '--reference_file', type=str, required=False,
                        default=r"C:\Users\tpdud\code\gogo\클러스터\reference_1.json",
                        help="Path to the pre-generated reference JSON file.")
    parser.add_argument('-d', '--embedding_directory', type=str, required=False,
                        default=r"C:\Users\tpdud\code\gogo\Database\summary\전처리 후",
                        help="Directory containing checkpoint embedding JSON files (e.g. checkpoint_2019.json, ...).")
    parser.add_argument('-m', '--metric', choices=['cosine', 'euclidean'], required=False, default='euclidean',
                        help="Distance metric for HDBSCAN clustering.")
    parser.add_argument('-c', '--cluster_selection_method', choices=["eom", "leaf"], required=False, default="eom",
                        help="Cluster selection method for HDBSCAN.")
    parser.add_argument('-s', '--min_cluster_size', type=int, required=False,
                        help="Minimum cluster size for HDBSCAN.")
    return parser

def main():
    args = create_parser().parse_args()
    
    reference_file = args.reference_file
    embedding_dir = args.embedding_directory
    metric = args.metric
    cluster_selection_method = args.cluster_selection_method
    min_cluster_size = args.min_cluster_size

    run_HDBSCAN(
        embedding_path=embedding_dir,
        reference_path=reference_file,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        min_cluster_size=min_cluster_size
    )

if __name__ == "__main__":
    main()
