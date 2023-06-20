#########################################################################################################################
##      IN ORDER TO EXECUTE THIS SCRIPT THAT IS IN CHARGE OF COLLECTING THE ALBUM_URIS FOR CERTAIN SONGS,              ##
##                                          SOME REQUIREMENTS ARE NEEDED.                                              ##
##                                                                                                                     ##
##      - SPOTIFY CREDENTIALS: Make sure you have valid spotify API credentials in the spotify_credentials.py file     ##
##      - INPUT DATASET: A dataset containing ['artistname', 'trackname'] columns and ['occurrences'] optional column. ##
##      - CONSTANTS: The DATASET_TAG and INPUT_PATH constants must be changed to run the new dataset.                  ##
##      - OUTPUT DATASET: The output dataset will contain the same input columns but adding a new column ['album_uri'] ##
##        referring to the uri of the album.                                                                           ##
##                                                                                                                     ##
#########################################################################################################################

import sys
import os
import pandas as pd
import spotipy
from tqdm import tqdm
from spotipy.oauth2 import SpotifyClientCredentials

import spotify_credentials
import constants

# get constants

#############################################
###### CHANGE DEPENDING ON THE DATASET ######
DATASET_TAG = constants.DATASET_TAG
INPUT_PATH = constants.PATH_TO_MSD_ORIGINAL 
#############################################

TRACK_URI = constants.TRACK_URI
OCC = constants.OCC
SAVING_STEP = constants.SAVING_STEP

OUTPUT_ALBUM_URIS = constants.ALBUM_URIS
CRAWLER_OUTPUT_FOLDER = constants.CRAWLER_OUTPUT_FOLDER


def obtain_album_uris(input_path=INPUT_PATH, index_from=0):

    print('Obtain album_uris based on track_names from: ', INPUT_PATH)
    tracks_df = pd.read_csv(INPUT_PATH, sep=',')

    # Instantiate Spotify client
    spotify = get_spotipy_client()
    print("client done")  

    if not os.path.exists(OUTPUT_ALBUM_URIS):
        album_df = pd.DataFrame({'trackname': pd.Series(dtype='str'), 'album': pd.Series(dtype='str'), 
                                 'album_uri': pd.Series(dtype='str'), 'occurrences': pd.Series(dtype='int')})
        # IMPORTANT: Occurrences quiza solo lo generamos antes en este caso concreto de Spotify Playlists, quiza habria que poner un IF
        index_from = 0
    else:
        album_df = pd.read_csv(OUTPUT_ALBUM_URIS, sep=',')
        if index_from == 0: index_from = len(album_df)


    for index, entry in tqdm(tracks_df.iterrows(), total=tracks_df.shape[0]):
        if index >= index_from:
            try:
                res = spotify.search(q=f"{entry['trackname']} {entry['artistname']}", limit=1, type='track')
                album_uri = res['tracks']['items'][0]['album']['uri']
                album_name = res['tracks']['items'][0]['album']['name']

                new_row = [entry['trackname'], album_name, album_uri, entry['occurrences']]
                album_df.loc[len(album_df)] = new_row

            except Exception as e:
                print('Trackname error', e)
                

        if index > index_from and index % SAVING_STEP == 0:
            album_df.to_csv(OUTPUT_ALBUM_URIS, index=False)


def get_spotipy_client():
    return spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=spotify_credentials.CLIENT_ID_5,
            client_secret=spotify_credentials.CLIENT_SECRET_5,
        )
    )


def main():
    if not os.path.exists(CRAWLER_OUTPUT_FOLDER):
        os.mkdir(CRAWLER_OUTPUT_FOLDER)
    obtain_album_uris()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        sys.exit(-1)

