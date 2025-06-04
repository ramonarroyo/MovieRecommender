import os
from django.core.management.base import BaseCommand
from src import IMDBDatasetDownloader, MovieDatasetReducer


class Command(BaseCommand):
    help = "Download IMDb datasets and generate reduced CSV file"

    def add_arguments(self, parser):
        parser.add_argument(
            '--percentage',
            type=float,
            default=0.90,
            help='Quantile of votes to keep (e.g. 0.90 keeps top 10%% of movies)'
        )
        parser.add_argument(
            '--output', default='movies_10.csv',
            help='Output CSV filename'
        )

    def handle(self, *args, **options):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
        downloader = IMDBDatasetDownloader(output_dir=base_dir)
        downloader.download()

        reducer = MovieDatasetReducer()
        output_name = os.path.splitext(options['output'])[0]
        os.chdir(base_dir)
        reducer.reduce_dataset(options['percentage'], output_name)
        self.stdout.write(self.style.SUCCESS(f'Dataset saved to {output_name}.csv'))
