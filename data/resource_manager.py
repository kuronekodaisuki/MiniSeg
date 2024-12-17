# -*- coding: shift_jis -*-
from data.common_define import ErrorCode

class ResourceManager:
    """Structure of a block information"""

    def __init__(self):
        self.resource_dict = {
            ErrorCode.LOAD_CONFIG_FAIL: "configsファイルのロードに失敗しました。",
            ErrorCode.DETECTION_CONFIG_INVALID: "検査パラメータが正しく設定されていません。",
            ErrorCode.INPUT_IMAGE_FOLDER_PATH_INCORRECT: "画像の入力保存フォルダーが正しく設定されていません。",
            ErrorCode.OUTPUT_IMAGE_FOLDER_PATH_INCORRECT: "画像の出力保存フォルダーが正しく設定されていません。",
            ErrorCode.CSV_FOLDER_PATH_INCORRECT: "csv保存フォルダーが正しく設定されていません。",
            ErrorCode.LOG_FOLDER_PATH_INCORRECT: "ログ保存フォルダーが正しく設定されていません。",
            ErrorCode.MODEL_NOT_EXIST: "modelファイルが見つかりません。",
            ErrorCode.LOAD_MODEL_FAIL: "",
            ErrorCode.READ_LINE_INFO_FAIL: "「%s」の読み込みに失敗しました。",
            ErrorCode.LINE_INFO_TEXT_NAME_INCORRECT: "「%s」のファイル名が正しくありません。",
            ErrorCode.TOTAL_BLOCK_INCORRECT: "「%s」のブロック数が正しく入力されていません。",
            ErrorCode.DETECTION_IMAGE_FORMAT_INCORRECT: "「%s」検査画像のファイル名が正しくありません。(_不正)",
            ErrorCode.DETECTION_IMAGE_DATETIME_INCORRECT: "「%s」検査画像のファイル名が正しくありません。(日付不正)",
            ErrorCode.DETECTION_IMAGE_BODY_NUMBER_INCORRECT: "「%s」検査画像のファイル名が正しくありません。(ボディ番号)",
            ErrorCode.DETECTION_AREA_IMAGE_FORMAT_INCORRECT: "「%s」検査領域画像のファイル名が正しくありません。(_不正)",
            ErrorCode.DETECTION_AREA_IMAGE_DATETIME_INCORRECT: "「%s」検査領域画像のファイル名が正しくありません。(日付不正)",
            ErrorCode.DETECTION_AREA_IMAGE_BODY_NUMBER_INCORRECT: "「%s」検査領域画像のファイル名が正しくありません。(ボディ番号)",
            ErrorCode.DETECTION_IMAGE_NOT_FOUND: "「%s」検査画像が存在しません。",
            ErrorCode.CANNOT_READ_IMAGE: "「%s」の画像の読み込みに失敗しました。",
            ErrorCode.SPLIT_IMAGE_FAIL: "「%s」の画像の分割に失敗しました。",
            ErrorCode.MASKING_IMAGE_FAIL: "「%s」のマスキングに失敗しました。",
            ErrorCode.DRAW_BOUND_FAIL: "「%s」のバウンディングボックスの描画に失敗しました。",
            ErrorCode.COMBINE_RESULT_IMAGE_FAIL: "「%s」の検査結果画像結合に失敗しました。",
            ErrorCode.DETECTION_NO_MASKING_FAIL: "「%s」のマスク無しの推論に失敗しました。",
            ErrorCode.DETECTION_MASKING_FAIL: "「%s」のマスク有りの推論に失敗しました。",
            ErrorCode.CREATE_OUTPUT_IMAGE_FOLDER_FAIL: "「%s」検査結果画像の保存フォルダ作成に失敗しました。",
            ErrorCode.WRITE_RESULT_IMAGE_FAIL: "「%s」検査結果画像の書き込みに失敗しました。",
            ErrorCode.CREATE_CSV_FAIL: "「%s」ｃｓｖファイル(一時)の作成に失敗しました。",
            ErrorCode.WRITE_DATA_CSV_FAIL: "「%s」ｃｓｖファイルの書き込みに失敗しました。",
            ErrorCode.MOVE_CSV_FAIL: "「%s」ｃｓｖファイルの移動に失敗しました。",
            ErrorCode.DELETE_IMAGE_FAIL: "「%s」検査対象ファイルの削除に失敗しました。",
            ErrorCode.WRITE_LOG_FAIL: "「%s」logファイルの書き込みに失敗しました。",
            ErrorCode.FILE_EXTENSION_INCORRECT: "「%s」ファイル拡張子が正しくありません。",
            ErrorCode.FILE_EXTENSION_INCORRECT_NOT_TIF: "「%s」「%s」ファイル拡張子が正しくありません。",
            ErrorCode.FILE_SIZE_ERROR: "「%s」「%s」ファイル容量が規定を超えています。",
            ErrorCode.DETECTION_CONFIG_SIZE_ERROR: "検査サイズが正しく設定されていません。",
            ErrorCode.DETECTION_CONFIG_SIZE_INVALID_VALUE: "検査サイズ閾値が指定可能範囲外です。",
        }

    def get_string(self, error_code):
        if self.resource_dict.get(error_code) is not None:
            return self.resource_dict.get(error_code)
        return ""
