import pandas as pd
import time

class EPGCleaner:
    """
    
    """

    def __init__(self,
                 make_collection: bool = True, 
                 merge_duplicated_desc: bool = True,
                 remove_empty_desc: bool = True,
                 verbose: bool = False,
                 ):
        self.merge_duplicated_desc = merge_duplicated_desc
        self.remove_empty_desc = remove_empty_desc
        self.verbose = verbose


    def clean(self, raw_data:pd.DataFrame) -> pd.DataFrame:
        """
        Clean the provided EPG DataFrame.

        Steps performed:
        1. Copy the original data to avoid modifying it in-place.
        2. Add a unique identifier (`files_id`) to each row for traceability.
        3. Optionally remove rows with empty descriptions.
        4. Optionally merge rows with identical descriptions while preserving 
        the first value of other columns and aggregating `files_id` as a tuple.
        5. Print detailed logs if verbose mode is enabled.

        Args:
            - raw_data (pd.DataFrame): The raw EPG data to clean.

        Returns 
            - pd.DataFrame
        """
        # Step 0:  Start the timer
        start = time.time()
        
        # Step 1: Work on a copy to keep the original DataFrame intact
        data = raw_data.copy()

        if self.verbose:
            total_rows = data.shape[0]
            missing_desc = data["desc"].isnull().sum()
            print(f"[Data Cleaning] Total rows: {total_rows}")
            print(f"[Data Cleaning] Rows with missing descriptions: {missing_desc} ({missing_desc / total_rows:.2%})")

        # Step 2: Add a unique ID to each row for traceability
        data["files_id"] = data.index
        
        # Step 3: Remove rows where the description is empty
        if self.remove_empty_desc or self.merge_duplicated_desc:
            data = data[data["desc"].notnull()]
            if self.verbose:
                print(f"[Data Cleaning] Removed rows with missing descriptions")
                print(f"[Data Cleaning] Remaining rows : {data.shape[0]}")
        
        # Step 4: Merge rows with identical descriptions
        if self.merge_duplicated_desc:
            pre_dedup_rows = data.shape[0]
            # Prepare aggregation rules: keep first value for all columns except 'files_id'
            columns = [col for col in data.columns if col not in ["desc", "files_id"]]
            agg_dict = {"files_id": tuple}
            for col in columns:
                agg_dict[col] = "first"
            # Perform groupby aggregation
            data = data.groupby("desc", as_index=False).agg(agg_dict)

            if self.verbose:
                post_dedup_rows = data.shape[0]
                removed_rows = pre_dedup_rows - post_dedup_rows
                print(f"[Data Cleaning] Aggregated duplicate descriptions")
                print(f"[Data Cleaning] Duplicates removed : {removed_rows} ({removed_rows / pre_dedup_rows:.2%})")

        if self.verbose:
            print(f"[Data Cleaning] Final dataset size: {data.shape[0]} rows ({(total_rows - data.shape[0]) / total_rows:.2%} reduction from original)")
            print(f"[Data Cleaning] Run in {time.time() - start:.2f} seconds")

        return data
            