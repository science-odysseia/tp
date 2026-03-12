import flet as ft
import flet_charts as fch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype


DATA_PATH = r"./student_dropout_dataset_v3.csv"

# 샘플 CSV 기준으로 범주형 컬럼 지정
CATEGORICAL_COLS = {
    "Gender",
    "Internet_Access",
    "Part_Time_Job",
    "Scholarship",
    "Semester",
    "Department",
    "Parental_Education",
}

# 굳이 분석할 필요 없는 ID 컬럼 제외
EXCLUDE_COLS = {"Student_ID"}


def main(page: ft.Page):
    page.title = "Student Dropout Analysis Dashboard"
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.Colors.GREY_100

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        page.add(ft.Text(f"데이터 파일을 찾을 수 없습니다.\n{e}", color=ft.Colors.RED))
        return

    # 분석 대상 컬럼만 사용
    df = df[[c for c in df.columns if c not in EXCLUDE_COLS]].copy()

    selectable_cols = list(df.columns)

    def col_kind(col_name: str) -> str:
        """
        반환값:
        - 'categorical'
        - 'binary'
        - 'numeric'
        """
        if col_name in CATEGORICAL_COLS:
            return "categorical"

        if is_numeric_dtype(df[col_name]):
            values = set(df[col_name].dropna().unique().tolist())
            if values.issubset({0, 1}) and len(values) <= 2:
                return "binary"
            return "numeric"

        return "categorical"


    def nice_step(min_val, max_val, target_ticks=5):
        span = float(max_val) - float(min_val)
        if span <= 0:
            return 1.0

        raw = span / target_ticks
        power = 10 ** np.floor(np.log10(raw))

        for mult in [1, 2, 5, 10]:
            step = mult * power
            if raw <= step:
                return float(step)

        return float(10 * power)

    def calc_simple_regression_stats(x_vals, y_vals):
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        n = len(x)

        if n < 2:
            return {"n": n, "r": np.nan, "r2": np.nan, "f": np.nan}

        if np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
            return {"n": n, "r": np.nan, "r2": np.nan, "f": np.nan}

        r = float(np.corrcoef(x, y)[0, 1])
        r2 = float(r ** 2)

        if n <= 2:
            f_val = np.nan
        elif np.isclose(1 - r2, 0):
            f_val = np.inf
        else:
            f_val = float((r2 / (1 - r2)) * (n - 2))

        return {"n": n, "r": r, "r2": r2, "f": f_val}

    def fmt_stat(v):
        if pd.isna(v):
            return "N/A"
        if np.isinf(v):
            return "∞"
        return f"{v:.4f}"

    def make_bar_chart(series: pd.Series, x_name: str, y_name: str):
        series = series.dropna()

        if series.empty:
            return ft.Text("표시할 데이터가 없습니다.", color=ft.Colors.RED)

        max_y = float(series.max()) * 1.15
        if max_y <= 0:
            max_y = 1.0
        if max_y < 1.0:
            max_y = 1.0

        y_step = nice_step(0, max_y, target_ticks=5)

        return fch.BarChart(
            interactive=True,
            min_y=0,
            max_y=max_y,
            border=ft.Border.all(1, ft.Colors.GREY_300),

            horizontal_grid_lines=fch.ChartGridLines(
                interval=y_step,
                color=ft.Colors.GREY_300,
                width=1,
                dash_pattern=[3, 3]
            ),
            vertical_grid_lines=fch.ChartGridLines(
                interval=1,
                color=ft.Colors.GREY_200,
                width=1,
                dash_pattern=[3, 3]
            ),

            left_axis=fch.ChartAxis(
                title=ft.Text(y_name, size=12, weight=ft.FontWeight.BOLD),
                title_size=32,
                show_min=True,
                show_max=True
            ),

            bottom_axis=fch.ChartAxis(
                title=ft.Text(x_name, size=12, weight=ft.FontWeight.BOLD),
                title_size=32,
                labels=[
                    fch.ChartAxisLabel(
                        value=i,
                        label=ft.Text(str(k), size=10)
                    )
                    for i, k in enumerate(series.index)
                ],
                label_size=70,
                show_min=True,
                show_max=True
            ),

            groups=[
                fch.BarChartGroup(
                    x=i,
                    rods=[
                        fch.BarChartRod(
                            from_y=0,
                            to_y=float(v),
                            width=24,
                            color=ft.Colors.BLUE_400
                        )
                    ]
                )
                for i, v in enumerate(series.values)
            ],
        )

    def make_line_chart(x_vals, y_vals, x_name: str, y_name: str):
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return ft.Text("회귀선을 그리기 위한 데이터가 부족합니다.", color=ft.Colors.RED)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)

        x_line = np.linspace(x.min(), x.max(), 60)
        y_line = model.predict(x_line.reshape(-1, 1))

        stats = calc_simple_regression_stats(x, y)

        if y_name == "Dropout":
            min_y = 0
            max_y = 1
        else:
            y_all_min = float(min(y.min(), y_line.min()))
            y_all_max = float(max(y.max(), y_line.max()))
            pad = max((y_all_max - y_all_min) * 0.15, 0.2)
            min_y = y_all_min - pad
            max_y = y_all_max + pad

        x_step = nice_step(float(x.min()), float(x.max()), target_ticks=6)
        y_step = nice_step(float(min_y), float(max_y), target_ticks=5)

        chart = fch.LineChart(
            interactive=True,
            min_x=float(x.min()),
            max_x=float(x.max()),
            min_y=float(min_y),
            max_y=float(max_y),
            border=ft.Border.all(1, ft.Colors.GREY_300),

            horizontal_grid_lines=fch.ChartGridLines(
                interval=y_step,
                color=ft.Colors.GREY_300,
                width=1,
                dash_pattern=[3, 3]
            ),
            vertical_grid_lines=fch.ChartGridLines(
                interval=x_step,
                color=ft.Colors.GREY_200,
                width=1,
                dash_pattern=[3, 3]
            ),

            left_axis=fch.ChartAxis(
                title=ft.Text(y_name, size=12, weight=ft.FontWeight.BOLD),
                title_size=32,
                show_min=True,
                show_max=True
            ),

            bottom_axis=fch.ChartAxis(
                title=ft.Text(x_name, size=12, weight=ft.FontWeight.BOLD),
                title_size=32,
                show_min=True,
                show_max=True
            ),

            data_series=[
                fch.LineChartData(
                    points=[
                        fch.LineChartDataPoint(float(px), float(py))
                        for px, py in zip(x, y)
                    ],
                    curved=True,
                    color=ft.Colors.GREEN_500,
                    stroke_width=3
                ),
                fch.LineChartData(
                    points=[
                        fch.LineChartDataPoint(float(px), float(py))
                        for px, py in zip(x_line, y_line)
                    ],
                    curved=False,
                    color=ft.Colors.BLACK,
                    stroke_width=3
                )
            ]
        )

        stats_view = ft.Row(
            [
                ft.Text(
                    f"회귀식: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}",
                    size=12,
                    color=ft.Colors.GREY_800
                ),
                ft.Text(f"R = {fmt_stat(stats['r'])}", size=12, color=ft.Colors.GREY_800),
                ft.Text(f"R² = {fmt_stat(stats['r2'])}", size=12, color=ft.Colors.GREY_800),
                ft.Text(f"F = {fmt_stat(stats['f'])}", size=12, color=ft.Colors.GREY_800),
                ft.Text(f"n = {stats['n']}", size=12, color=ft.Colors.GREY_800),
            ],
            wrap=True,
            spacing=18,
        )

        return ft.Column([stats_view, chart], spacing=10)

    def make_crosstab_table(ct_percent: pd.DataFrame):
        columns = [ft.DataColumn(ft.Text("항목"))]
        columns += [ft.DataColumn(ft.Text(str(col))) for col in ct_percent.columns]

        rows = []
        for idx, row in ct_percent.iterrows():
            cells = [ft.DataCell(ft.Text(str(idx)))]
            for val in row:
                cells.append(ft.DataCell(ft.Text(f"{val:.1f}%")))
            rows.append(ft.DataRow(cells=cells))

        return ft.Column(
            [
                ft.Text("행 기준 비율(%)", size=14, weight=ft.FontWeight.BOLD),
                ft.DataTable(columns=columns, rows=rows)
            ],
            spacing=10
        )

    result_area = ft.Container()

    x_dropdown = ft.Dropdown(
        label="X 변수 선택",
        width=300,
        options=[ft.dropdown.Option(col) for col in selectable_cols],
        value="Attendance_Rate" if "Attendance_Rate" in selectable_cols else selectable_cols[0]
    )

    y_dropdown = ft.Dropdown(
        label="Y 변수 선택",
        width=300,
        options=[ft.dropdown.Option(col) for col in selectable_cols],
        value="Dropout" if "Dropout" in selectable_cols else selectable_cols[1]
    )

    def update_chart(e=None):
        x_col = x_dropdown.value
        y_col = y_dropdown.value

        if not x_col or not y_col:
            result_area.content = ft.Text("X, Y 변수를 모두 선택하세요.", color=ft.Colors.RED)
            page.update()
            return

        if x_col == y_col:
            result_area.content = ft.Text("서로 다른 변수를 선택하세요.", color=ft.Colors.RED)
            page.update()
            return

        sub = df[[x_col, y_col]].dropna().copy()

        if sub.empty:
            result_area.content = ft.Text("결측치를 제거하고 나니 표시할 데이터가 없습니다.", color=ft.Colors.RED)
            page.update()
            return

        x_kind = col_kind(x_col)
        y_kind = col_kind(y_col)

        info_lines = [
            ft.Text(f"선택 변수: {x_col} vs {y_col}", size=22, weight=ft.FontWeight.BOLD),
            ft.Text(f"X 타입: {x_kind} / Y 타입: {y_kind}", size=13, color=ft.Colors.GREY_700),
            ft.Text(f"사용 행 수: {len(sub)}", size=13, color=ft.Colors.GREY_700),
            ft.Divider()
        ]

        # 1) 범주형 X, 수치형/이진 Y
        if x_kind == "categorical" and y_kind in {"numeric", "binary"}:
            grp = sub.groupby(x_col)[y_col].mean().sort_values(ascending=False)
            chart = make_bar_chart(grp, x_col, f"{y_col} 평균")

            content = ft.Column(
                info_lines + [
                    ft.Text(f"{x_col} 범주별 {y_col} 평균", size=14),
                    chart
                ],
                spacing=10
            )

        # 2) 수치형/이진 X, 수치형/이진 Y
        elif x_kind in {"numeric", "binary"} and y_kind in {"numeric", "binary"}:
            unique_count = sub[x_col].nunique()

            if unique_count > 12:
                sub["x_bin"] = pd.cut(sub[x_col], bins=10, duplicates="drop")
                agg = sub.groupby("x_bin", observed=False)[y_col].mean().reset_index()
                agg = agg.dropna()

                if agg.empty:
                    result_area.content = ft.Text("구간화 후 표시할 데이터가 없습니다.", color=ft.Colors.RED)
                    page.update()
                    return

                agg["x_mid"] = agg["x_bin"].apply(lambda iv: (iv.left + iv.right) / 2)
                chart = make_line_chart(agg["x_mid"], agg[y_col], x_col, y_col)
                desc = f"{x_col}를 10개 구간으로 나누어 {y_col} 평균과 회귀선을 표시했습니다."
            else:
                agg = sub.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col)
                chart = make_line_chart(agg[x_col], agg[y_col], x_col, y_col)
                desc = f"{x_col} 값별 {y_col} 평균과 회귀선을 표시했습니다."

            content = ft.Column(
                info_lines + [
                    ft.Text(desc, size=14),
                    chart
                ],
                spacing=10
            )

        # 3) 수치형/이진 X, 범주형 Y
        elif x_kind in {"numeric", "binary"} and y_kind == "categorical":
            grp = sub.groupby(y_col)[x_col].mean().sort_values(ascending=False)
            chart = make_bar_chart(grp, x_col, f"{y_col} 평균")

            content = ft.Column(
                info_lines + [
                    ft.Text(f"{y_col} 범주별 {x_col} 평균", size=14),
                    chart
                ],
                spacing=10
            )

        # 4) 범주형 X, 범주형 Y
        else:
            ct_percent = pd.crosstab(sub[x_col], sub[y_col], normalize="index") * 100
            chart = make_crosstab_table(ct_percent)

            content = ft.Column(
                info_lines + [
                    ft.Text("두 범주형 변수 조합이므로 교차표로 표시했습니다.", size=14),
                    chart
                ],
                spacing=10
            )

        result_area.content = ft.Container(
            padding=20,
            border_radius=12,
            bgcolor=ft.Colors.WHITE,
            content=content
        )

        page.update()

    x_dropdown.on_change = update_chart
    y_dropdown.on_change = update_chart

    page.add(
        ft.Text(
            "Student Dropout Analysis Dashboard",
            size=30,
            weight=ft.FontWeight.BOLD
        ),
        ft.Text(
            "원하는 두 변수를 선택하면 자동으로 관계를 요약합니다.",
            size=14,
            color=ft.Colors.GREY_700
        ),
        ft.Row(
            [
                x_dropdown,
                y_dropdown,
                ft.ElevatedButton("분석하기", on_click=update_chart)
            ],
            wrap=True
        ),
        result_area
    )

    update_chart()


ft.run(main)
