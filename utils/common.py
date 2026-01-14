
import os
import json
import pickle
import shutil
import random
import re

class Helper:

    @staticmethod
    def read_from_json(file_path):
        """Reads data from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from: {file_path}.")
        return None

    @staticmethod
    def write_to_json(data, file_path):
        """Writes data to a JSON file."""
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error writing to JSON file {file_path}: {e}")

    @staticmethod
    def read_from_pickle(file_path):
        """Reads data from a pickle file."""
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}.")
        except pickle.UnpicklingError as e:
            print(f"Error reading from pickle file {file_path}: {e}")
        return None

    @staticmethod
    def write_to_pickle(data, file_path):
        """Writes data to a pickle file."""
        try:
            with open(file_path, "wb") as file:
                pickle.dump(data, file)
        except Exception as e:
            print(f"Error writing to pickle file {file_path}: {e}")


    @staticmethod
    def get_video_path(path: str) -> str:
        """Return a video path from file or folder; raise if none/multiple found."""
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            vids = [f for f in os.listdir(path) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
            if len(vids) == 1: return os.path.join(path, vids[0])
            if not vids: raise FileNotFoundError(f"No videos in {path}")
            raise ValueError(f"Multiple videos in {path}")
        raise FileNotFoundError(f"Path not found: {path}")

    @staticmethod
    def get_immediate_folder_name(path):
        """Returns the immediate folder name from a given file or directory path.

        Args:
            path: The path to a file or directory.

        Returns:
            The name of the immediate folder containing the file or directory.

        Raises:
            ValueError: If the provided path does not exist or is not a valid file or directory.
        """
        # Check if the provided path exists
        if not os.path.exists(path):
            raise ValueError(f"The provided path '{path}' does not exist.")
        
        # If the path is a file, get the directory name
        if os.path.isfile(path):
            folder_path = os.path.dirname(path)
        else:
            folder_path = path  # Treat the path as a directory if it is not a file

        # Get the immediate folder name
        folder_name = os.path.basename(folder_path.rstrip("/\\"))
        
        return folder_name

    @staticmethod
    def create_folder(folder_path):
        """Creates a folder if it does not already exist.

        Args:
            folder_path: The path of the folder to create.
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating folder {folder_path}: {e}")

    @staticmethod
    def create_subfolders(directory_path, folder_list=["images", "labels"]):
        """Creates specified subfolders within the provided directory (default: 'images' and 'labels').

        If the subfolder already exists, it will be deleted and recreated.

        Args:
            directory_path: The path of the directory
            folder_list: A list of subfolder names to create within the main directory. Defaults to ["images", "labels"].

        Returns:
            A list of paths to the created subfolders.
        """
        try:
            # Remove the main directory if it already exists and recreate it
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
            os.makedirs(directory_path, exist_ok=True)
        
            created_folder_paths = []
            for folder_name in folder_list:
                folder_path = os.path.join(directory_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                created_folder_paths.append(folder_path)

            return created_folder_paths
        except Exception as e:
            print(f"Error creating subfolders in {directory_path}: {e}")
            return []

    @staticmethod
    def get_subfolder_names(directory_path):
        """Returns the names of all subfolders in the given directory.

        Args:
            directory_path: The path of the directory to search.

        Returns:
            A list of subfolder names.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"The provided path '{directory_path}' is not a valid directory.")

        subfolders = []
        for name in os.listdir(directory_path):
            folder_path = os.path.join(directory_path, name)
            if os.path.isdir(folder_path):
                subfolders.append(name)

        return subfolders

    @staticmethod
    def get_subfolder_paths(directory_path, folder_list=["images", "labels"]):
        """Returns paths of specified subfolders in a directory.

        Args:
            directory_path: The path of the directory to search.
            folder_list: A list of folder names to return paths for. If `None`, all subfolder paths are returned. Defaults to ["images", "labels"].

        Returns:
            A list of paths to the specified subfolders, or all subfolders if `folder_list` is `None`.
        """
        try:
            if folder_list is None:
                # Return paths for all subfolders
                folder_list = Helper.get_subfolder_names(directory_path)
                
            # Return paths for specified subfolders
            return [os.path.join(directory_path, folder_name) for folder_name in folder_list]
        except Exception as e:
            print(f"Error retrieving subfolder paths in {directory_path}: {e}")
            return []

    @staticmethod
    def get_subfolders_with_prefixes(dir_path, prefixes):
        """
        Returns a list of subfolder names in the given directory 
        that start with any of the provided prefixes.

        :param dir_path: Path to the directory
        :param prefixes: List of prefixes to filter subfolders
        :return: List of matching subfolder names
        """
        subfolders = []
        if not os.path.isdir(dir_path):
            raise ValueError(f"{dir_path} is not a valid directory")

        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)
            if os.path.isdir(full_path) and any(name.startswith(p) for p in prefixes):
                subfolders.append(name)

        return subfolders

    @staticmethod
    def find_files_with_filter(folder, extension=".mp4", substring="", debug=True):
        """
        Search for files in a folder with a given extension and optional substring filter.
        Optionally prints matching files with serial numbers.

        Parameters:
            folder (str): Directory path to search in.
            extension (str): File extension to filter by (e.g., '.mp4', '.mp3').
            substring (str): Substring that must appear in the filename (case-insensitive).
            debug (bool): If True, print matching files. Default = True.

        Returns:
            list: List of matching file paths.
        """
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Provided path is not a directory: {folder}")

        extension = extension.lower()
        substring = substring.lower()

        matching_files = []
        for file in os.listdir(folder):
            if file.lower().endswith(extension) and substring in file.lower():
                matching_files.append(file)

        # Print results only if debug=True
        if debug:
            if matching_files:
                print(f"\nFound {len(matching_files)} file(s) matching "
                    f"extension='{extension}' and substring='{substring}':")
                for i, f in enumerate(sorted(matching_files), start=1):
                    print(f"{i}. {os.path.basename(f)}")
            else:
                print(f"\nNo files found in '{folder}' with extension='{extension}' "
                    f"and substring='{substring}'")

        return sorted(matching_files)



    @staticmethod
    def get_image_files(folder_path):
        """Retrieves a list of image files in the specified folder.
        
        Args:
            folder_path (str): Path of the folder to search for images.
        
        Returns:
            list: List of image file names in the folder, or an empty list if the folder doesn't exist.
        """
        image_extensions = (".jpg", ".jpeg", ".png")
        if os.path.exists(folder_path):
            return [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and file.lower().endswith(image_extensions)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []
    
    @staticmethod
    def get_sorted_frame_filenames(folder_path):
        """
        Reads a folder and returns a list of frame filenames sorted by timestamp.

        Args:
            folder_path (str): Path to the directory containing frame PNGs.

        Returns:
            list[str]: Sorted list of filenames (not full paths).
        """
        def parse_frame_filename(filename):
            """
            Extract (minutes, seconds, milliseconds) as integers from a filename.
            Example: frame_002.56.000.png -> (2, 56, 0)
            """
            match = re.match(r"frame_(\d{3})\.(\d{2})\.(\d{3})", filename)
            if not match:
                raise ValueError(f"Invalid filename format: {filename}")
            return tuple(map(int, match.groups()))

        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        sorted_files = sorted(all_files, key=parse_frame_filename)
        return sorted_files

    @staticmethod
    def get_files_with_extension(folder_path, extension):
        """Retrieves a list of files with a specific extension in the given folder.
        
        Args:
            folder_path (str): Path of the folder to search.
            extension (str): File extension to filter by (e.g., '.txt').
        
        Returns:
            list: List of files with the given extension, or an empty list if the folder doesn't exist.
        """
        if os.path.exists(folder_path):
            return [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and file.lower().endswith(extension)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []
    
    @staticmethod
    def get_files_with_extensions(folder_path, extensions):
        """Retrieves a list of files with specific extensions in the given folder.
        
        Args:
            folder_path (str): Path of the folder to search.
            extensions (tuple or list): File extensions to filter by.
        
        Returns:
            list: List of files matching the given extensions, or an empty list if the folder doesn't exist.
        """
        if os.path.exists(folder_path):
            if isinstance(extensions, list):
                extensions = tuple(extensions)
            return [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and file.lower().endswith(extensions)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []
    
    @staticmethod
    def get_files_without_extension(folder_path, extension):
        """Retrieves a list of file names (without extensions) that match a given extension.
        
        Args:
            folder_path (str): Path of the folder to search.
            extension (str): File extension to filter by (e.g., '.txt').
        
        Returns:
            list: List of file names without extensions, or an empty list if the folder doesn't exist.
        """
        if os.path.exists(folder_path):
            return [os.path.splitext(file)[0] for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)) and file.lower().endswith(extension)]
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return []



        
    @staticmethod
    def delete_files_withExtension(folder_path, extension):
        """Deletes all files with the specified extension in the given folder.

        Args:
            folder_path: The path of the folder from which to delete files.
            extension: The file extension to filter by (e.g., '.txt').
        """
        file_list = Helper.get_files_with_extension(folder_path, extension)
        for filename in file_list:
            filepath = os.path.join(folder_path, filename)
            os.remove(filepath)

    @staticmethod
    def delete_all_files_in_folder(folder_path):
        """Deletes all files and subfolders in the specified folder.

        Args:
            folder_path: The path of the folder from which to delete all files and subfolders.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)         # Delete the file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)     # Delete the subfolder and its contents
        except Exception as e:
            print(f"Error deleting files in '{folder_path}': {e}")




    @staticmethod
    def copy_specified_files(src_folder, dest_folder, file_list):
        """
        Copies specified files from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            file_list (list): List of filenames to copy.

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Ensure the destination folder exists
        Helper.create_folder(dest_folder)

        count = 0
        for filename in file_list:
            src_filepath = os.path.join(src_folder, filename)

            # Check if the file exists in the source folder
            if not os.path.isfile(src_filepath):
                print(f"File '{filename}' does not exist in the source folder.")
                continue

            try:
                shutil.copy(src_filepath, dest_folder)
                count += 1
            except OSError as e:
                print(f"Error copying '{filename}': {e}")

        return count

    @staticmethod
    def copy_specified_files_withExtension(src_folder, dest_folder, file_list, extension):
        """
            Copies files with the specified extension from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            file_list (list): List of filenames (without extensions) to copy.
            extension (str): File extension to append when looking for files.

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Ensure the destination folder exists
        Helper.create_folder(dest_folder)

        # Iterate over the file list and copy each file
        count = 0
        for filename in file_list:
            src_file_path = os.path.join(src_folder, filename + extension)

            # Check if the file exists in the source folder
            if not os.path.isfile(src_file_path):
                print(f"File '{filename + extension}' does not exist in the source folder.")
                continue

            try:
                shutil.copy(src_file_path, dest_folder)
                count += 1
            except OSError as e:
                print(f"Error copying '{filename + extension}': {e}")

        return count

    @staticmethod
    def copy_files_withExtension(src_folder, dest_folder, extension='.jpg'):
        """
        Copies files with a specified extension from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            extension (str): The file extension to filter files by (default is '.jpg').

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Get a list of files with the specified extension
        file_list = Helper.get_files_with_extension(src_folder, extension)

        # Copy the specified files to the destination folder
        Helper.copy_specified_files(src_folder, dest_folder, file_list)
        return len(file_list)

    @staticmethod
    def copy_files_withExtensions(src_folder, dest_folder, extensions):
        """
        Copies files with a list of specified extensions from the source folder to the destination folder.

        Args:
            src_folder (str): The source folder path.
            dest_folder (str): The destination folder path.
            extension (list or tuple): A tuple or list of file extensions to filter by.

        Raises:
            OSError: If there is an error during the file copy process.
        """
        # Get a list of files with the specified extension
        file_list = Helper.get_files_with_extensions(src_folder, extensions)

        # Copy the specified files to the destination folder
        Helper.copy_specified_files(src_folder, dest_folder, file_list)
        return len(file_list)
