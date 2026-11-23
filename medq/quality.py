"""إصلاحات جودة البيانات (الشخص 2). الآن نسخة وهمية ترجع الجدول كما هو."""
import pandas as pd


class QualityFixer:
    def apply(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """is_train=True: مسموح الحذف والتوحيد. is_train=False (test): ممنوع حذف أي صف."""
        return df.copy()
