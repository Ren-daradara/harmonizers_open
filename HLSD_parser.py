import json
import os, sys
import random
import argparse
from glob import glob

'''
Hook Lead Sheet Dataset
- Ref: https://github.com/wayne391/lead-sheet-dataset

** Save the original dataset at the directory 
   where this code belongs to --> /(codeDiretory)/HLSD/dataset/event/a/...
   元のデータセットをディレクトリに保存する 
   このコードが属するディレクトリ --> /(codeDiretory)/HLSD/dataset/event/a/...
'''
file_path = './HLSD/dataset/event/'#eventまで
result_path = './HLSD/output/'#output

def remove_ds(l):
    ds = '.DS_Store'
    if ds in l:
        l.remove(ds)
    return l

def replace(path):
    return result_path + '/'.join(path.split('/')[3:])

def parse_with_args(args):
    with open(args.file_dir, "r") as fr:
        illegal = False
        not_fit_cnt = 0

        dct = json.load(fr)
        aligned_dct = {}

        bim = dct['metadata']['beats_in_measure']
        if bim != '4':
            print('parse failed: the song does not have 4 beat time signature.')
            return
        if len(dct['tracks']['melody']) == 0 or len(dct['tracks']['chord']) == 0:
            print('parse failed: the song has either empty melody or empty chord.')
            return

        aligned_dct['time_signature'] = dct['metadata']['beats_in_measure']
        aligned_dct['key_signature'] = dct['metadata']['key']
        mode = dct['metadata']['mode']
        if mode == '1':
            aligned_dct['mode'] = 'major'
        elif mode == '6':
            aligned_dct['mode'] = 'minor'
        else:
            print(f'{args.file_dir}: mode {mode}')
            aligned_dct['mode'] = 'unknown'

        melody = dct['tracks']['melody']
        aligned_melodies = []
            
        prev_end = 0.0
        # initialize aligned melody dictionary
        for idx, m in enumerate(melody):
            if m is None:
                continue
            aligned_m = {}
            if m['event_on'] != prev_end:
                aligned_m['pitch'] = 0
                aligned_m['is_rest'] = True
                aligned_m['start'] = prev_end
                aligned_m['end'] = m['event_on']
                aligned_melodies.append(aligned_m)
                aligned_m = {}
            aligned_m['pitch'] = m['pitch'] + 48
            aligned_m['is_rest'] = m['isRest']
            aligned_m['start'] = m['event_on']
            aligned_m['end'] = m['event_off']
            prev_end = m['event_off']

            aligned_melodies.append(aligned_m.copy())

        aligned_dct['melody'] = aligned_melodies

        chord = dct['tracks']['chord']
        aligned_chords = []

        empty_chord = 0

        beat_cnt = 4 * dct['num_measures']
        for i in range(int(beat_cnt)):
            start, end = i, i+1
            fit = False
            aligned_c = {}
            aligned_c['start'] = start
            aligned_c['end'] = end
            for c in chord:
                if c is None:
                    continue
                if c['event_on'] <= start and end <= c['event_off']:
                    fit = True
                    aligned_c['composition'] = c['composition']
                    break
            if not fit:
                not_fit_cnt += 1
                aligned_c_list = []
                for c in chord:
                    if c is None:
                        continue
                    if start <= c['event_on'] <= end or start <= c['event_off'] <= end:
                        aligned_c_list.append(c['composition'])
                if len(aligned_c_list) == 0:
                    empty_chord += 1
                    continue
                aligned_c['composition'] = random.choice(aligned_c_list)
            aligned_c['composition'] = sorted(list(map(lambda x: x % 12, aligned_c['composition'])))
            aligned_chords.append((aligned_c.copy()))
        
        if empty_chord != 0:
            illegal = True
        aligned_dct['chord'] = aligned_chords

        # after parsing and aligning
        if not illegal:
            if not os.path.isdir('/'.join((args.out_dir + '/' + args.file_dir).split('/')[:-1])):
                os.makedirs('/'.join((args.out_dir + '/' + args.file_dir).split('/')[:-1]))
            with open(args.out_dir + '/' + args.file_dir, 'w') as fw:
                json.dump(aligned_dct, fw)
                print(f'{args.file_dir} successfully parsed at {args.out_dir}')

def parse(file_dir):
    class Args:
        def __init__(self, fd):
            self.file_dir = fd
            self.out_dir = 'output'
    args = Args(file_dir)
    parse_with_args(args)

def run_all():
    # primary subdirectories of HTD/event is alphabetical indices, followed by names of artists.
    #アルファベット順とそれに続いて、アーティスト名が続く
    #alpha_index = list(map(lambda x: chr(x+ord('a')), range(26)))
    #
    #alpha_file_path = list(map(lambda x: file_path+x+'/', alpha_index))
    alpha_file_path = sorted(glob(os.path.join(file_path, '*/')))
    #全アルファベットのファイルを収集

    # 全楽曲数
    total_cnt = 0
    # 解析出来た楽曲の数
    file_cnt = 0
    # number of songs without 4 beat time signatures (not parsed)4/4拍子でない曲の数
    not_four_cnt = 0
    # number of songs with empty melody or empty chords (not parsed)メロディとコードがからの曲
    empty_mc_cnt = 0
    # number of songs with empty chords (not parsed)コードが空の曲
    illegal_cnt = 0
    # number of songs with unaligned chord for melody (not parsed)整列されていない曲の数？
    unaligned_songs = 0

    major_cnt, minor_cnt, unknown_cnt = 0, 0, 0

    for al in alpha_file_path:#全アルファベットのデータに対して
        if not os.path.exists(replace(al)):#もしoutputにパスが存在していなければ
            os.makedirs(replace(al))#output様のパスを作る
        artists = remove_ds(os.listdir(al))
        #アルファベット名以下のファイルディレクトリ一覧を取得
        artist_path = list(map(lambda x: al + x + '/', artists))
        #アルファベット名+x+/の処理をartists全体に行う
        for artist in artist_path:#アーティスト名のファイルに対して
            if not os.path.exists(replace(artist)):
                os.makedirs(replace(artist))#outputにファイル名がなければ作成する
            songs = remove_ds(os.listdir(artist))
            song_path = list(map(lambda x: artist + x + '/', songs))
            for song in song_path:#全ての楽曲名のファイルに対して
                if not os.path.exists(replace(song)):
                    os.makedirs(replace(song))#outputに無ければ作成
                files = remove_ds(os.listdir(song))
                files = list(filter(lambda x: x.endswith('_key.json'), files))
                #_key.jsonで終わるファイルのみをいれる
                for f in files:#全てのパートファイルについて
                    total_cnt += 1
                    with open(song + f, "r") as fr:
                        illegal = False
                        not_fit_cnt = 0

                        dct = json.load(fr)#jsonファイル読み込み
                        aligned_dct = {}

                        bim = dct['metadata']['beats_in_measure']
                        if bim != '4':#4/4でないと飛ばす
                            not_four_cnt += 1
                            continue
                        if len(dct['tracks']['melody']) == 0 or len(dct['tracks']['chord']) == 0:
                            empty_mc_cnt += 1
                            continue#melodyの長さが0もしくはchordの長さが0でも飛ばす
        
                        aligned_dct['time_signature'] = dct['metadata']['beats_in_measure']
                        #4/4拍子であることを入力
                        aligned_dct['key_signature'] = dct['metadata']['key']
                        #key情報を入力C,D...
                        mode = dct['metadata']['mode']
                        if mode == '1':
                            aligned_dct['mode'] = 'major'
                            major_cnt += 1
                        elif mode == '6':
                            aligned_dct['mode'] = 'minor'
                            minor_cnt += 1
                        else:
                            print(f'{song + f}: mode {mode}')
                            aligned_dct['mode'] = 'unknown'
                            unknown_cnt += 1
                        #modeをmajor,minor,unknownでタグ付け

                        melody = dct['tracks']['melody']
                        aligned_melodies = []
                            
                        prev_end = 0.0
                        # initialize aligned melody dictionary
                        for idx, m in enumerate(melody):
                            #辞書に入っていた全てのmelodyに対して
                            if m is None:
                                continue
                            aligned_m = {}#初期化
                            if m['event_on'] != prev_end:#event_onが前音のoffsetじゃない場合
                                aligned_m['pitch'] = 0#ピッチ0
                                aligned_m['is_rest'] = True#is_rest=True
                                aligned_m['start'] = prev_end#startは前回の終わり
                                aligned_m['end'] = m['event_on']#offは今回の始まり
                                aligned_melodies.append(aligned_m)#上記の辞書を追加
                                aligned_m = {}#初期化
                            aligned_m['pitch'] = m['pitch'] + 48#ピッチ+48
                            aligned_m['is_rest'] = m['isRest']#isRest
                            aligned_m['start'] = m['event_on']#event_on
                            aligned_m['end'] = m['event_off']#event_off
                            prev_end = m['event_off']#prev_endの更新

                            aligned_melodies.append(aligned_m.copy())#コピーを追加

                        aligned_dct['melody'] = aligned_melodies#集めたmelodyを辞書に追加

                        chord = dct['tracks']['chord']#chordから辞書を取ってくる
                        aligned_chords = []#初期化

                        empty_chord = 0

                        beat_cnt = 4 * dct['num_measures']
                        for i in range(int(beat_cnt)):#全ての拍に対して
                            start, end = i, i+1
                            fit = False
                            aligned_c = {}
                            aligned_c['start'] = start
                            aligned_c['end'] = end
                            for c in chord:#全てのコードに対して
                                if c is None:
                                    continue#コードが無ければ続ける
                                if c['event_on'] <= start and end <= c['event_off']:
                                    #event_onがstartより早く、event_offがendより遅い場合
                                    fit = True
                                    aligned_c['composition'] = c['composition']
                                    #そのコードのcompositionを辞書に送る
                                    break
                            if not fit:#上記でコードが見つからなかった場合
                                not_fit_cnt += 1
                                aligned_c_list = []
                                for c in chord:#全てのコードに対して
                                    if c is None:#コードが無ければ飛ばす
                                        continue
                                    if start <= c['event_on'] <= end or start <= c['event_off'] <= end:
                                        #該当する拍内に被るコード進行があればリストに入れる
                                        aligned_c_list.append(c['composition'])
                                if len(aligned_c_list) == 0:
                                    #拍内に該当するコードが無ければ飛ばす
                                    empty_chord += 1
                                    continue
                                aligned_c['composition'] = random.choice(aligned_c_list)
                                #リストからランダムにコード進行を選択
                            aligned_c['composition'] = sorted(list(map(lambda x: x % 12, aligned_c['composition'])))
                            #compositionを12で割った余りの値をリストに入れる
                            aligned_chords.append((aligned_c.copy()))

                        
                        if empty_chord != 0:#空のコードがあった場合
                            illegal_cnt += 1
                            illegal = True
                        if not illegal and not_fit_cnt != 0:
                            #空のコードはないが、コードが一つも無いとき
                            unaligned_songs += 1
                        aligned_dct['chord'] = aligned_chords


                        # after parsing and aligning
                        if not illegal:#空のコードが無いとき
                            with open(replace(song+f), 'w') as fw:
                                json.dump(aligned_dct, fw)
                                file_cnt += 1
                                if file_cnt % 100 == 0:
                                    print(f'{file_cnt} files parsed')
                                    #辞書に変換してoutputに保存
    # remove empty directories recursively
    def remove_r(path):
        if not os.path.isdir(path):
            return
        files = os.listdir(path)
        if len(files):
            for f in files:
                fullpath = os.path.join(path, f)
                if os.path.isdir(fullpath):
                    remove_r(fullpath)

        files = os.listdir(path)
        if len(files) == 0:
            os.rmdir(path)

    remove_r(result_path)
    print(f'total songs: {total_cnt}')
    print(f'songs without 4 beat time signature: {not_four_cnt}')
    print(f'songs with empty melody or empty chord: {empty_mc_cnt}')
    print(f'songs with empty chord: {illegal_cnt}')
    print('-'*50)
    print(f'parsed songs: {file_cnt}')
    print(f'songs with unaligned beats: {unaligned_songs}')
    print(f'number of mode: {major_cnt} majors, {minor_cnt} minors, {unknown_cnt} unknown')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', dest='file_dir', default='all')#プログラム実行時にfile_dirの名前の引数を設定するオプション省略時はall
    parser.add_argument('--out_dir', dest='out_dir', default='output')#上記同様省略時はout_dir=output
    args = parser.parse_args()
    if args.file_dir == 'all':
        run_all()
    else:
        parse_with_args(args)
