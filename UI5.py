import flet as ft
import flet_charts as fch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype

DATA_PATH = r"./student_dropout_dataset_v3.csv"

CATEGORICAL_COLS = {
    "Gender", "Internet_Access", "Part_Time_Job", 
    "Scholarship", "Semester", "Department", "Parental_Education",
}

EXCLUDE_COLS = {"Student_ID"}

def main(page: ft.Page):
    page.title = "Student Dropout Analysis Dashboard"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.Colors.GREY_100

    try:
        df_origin = pd.read_csv(DATA_PATH)
    except Exception as e:
        page.add(ft.Text(f"데이터 파일을 찾을 수 없습니다.\n{e}", color=ft.Colors.RED))
        return

    df = df_origin[[c for c in df_origin.columns if c not in EXCLUDE_COLS]].copy()

    def col_kind(col_name: str) -> str:
        if col_name in CATEGORICAL_COLS: return "categorical"
        if is_numeric_dtype(df[col_name]):
            values = set(df[col_name].dropna().unique().tolist())
            if values.issubset({0, 1}) and len(values) <= 2: return "binary"
            return "numeric"
        return "categorical"

    def nice_step(min_val, max_val, target_ticks=5):
        span = float(max_val) - float(min_val)
        if span <= 0: return 1.0
        raw = span / target_ticks
        power = 10 ** np.floor(np.log10(raw))
        for mult in [1, 2, 5, 10]:
            step = mult * power
            if raw <= step: return float(step)
        return float(10 * power)

    def calc_simple_regression_stats(x_vals, y_vals):
        x, y = np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 2 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
            return {"n": n, "r": np.nan, "r2": np.nan, "f": np.nan}
        r = float(np.corrcoef(x, y)[0, 1]); r2 = float(r ** 2)
        f_val = np.nan if n <= 2 else (np.inf if np.isclose(1 - r2, 0) else float((r2 / (1 - r2)) * (n - 2)))
        return {"n": n, "r": r, "r2": r2, "f": f_val}

    def fmt_stat(v):
        if pd.isna(v): return "N/A"
        if np.isinf(v): return "∞"
        return f"{v:.4f}"

    def make_bar_chart(series, x_name, y_name, bar_color=ft.Colors.BLUE_400):
        series = series.dropna()
        if series.empty: return ft.Text("표시할 데이터가 없습니다.")
        max_y = float(series.max()) * 1.2
        if max_y < 1.0: max_y = 1.0

        return fch.BarChart(
            interactive=True, min_y=0, max_y=max_y,
            border=ft.Border.all(1, ft.Colors.GREY_300),
            left_axis=fch.ChartAxis(title=ft.Text(y_name, size=10, weight="bold"), title_size=50, label_size=50),
            bottom_axis=fch.ChartAxis(title=ft.Text(x_name, size=10, weight="bold"), title_size=32, label_size=30,
                labels=[fch.ChartAxisLabel(value=i, label=ft.Text(str(k), size=8)) for i, k in enumerate(series.index)]),
            groups=[
                fch.BarChartGroup(
                    x=i, 
                    rods=[
                        fch.BarChartRod(
                            from_y=0, to_y=float(v), width=25, color=bar_color, 
                            border_radius=ft.BorderRadius.only(top_left=6, top_right=6),
                            tooltip=f"{v*100:.1f}%"
                        )
                    ]
                ) for i, v in enumerate(series.values)
            ],
        )

    def make_line_chart(x_vals, y_vals, x_name, y_name):
        x, y = np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
        if len(x) < 2: return ft.Text("데이터 부족")
        order = np.argsort(x); x, y = x[order], y[order]
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        x_line = np.linspace(x.min(), x.max(), 60); y_line = model.predict(x_line.reshape(-1, 1))
        stats = calc_simple_regression_stats(x, y)
        min_y, max_y = (0, 1) if y_name == "자퇴율" else (float(min(y.min(), y_line.min())*0.9), float(max(y.max(), y_line.max())*1.1))
        chart = fch.LineChart(
            interactive=True, min_x=float(x.min()), max_x=float(x.max()), min_y=float(min_y), max_y=float(max_y),
            border=ft.Border.all(1, ft.Colors.GREY_300),
            left_axis=fch.ChartAxis(title=ft.Text(y_name, size=10, weight="bold"), title_size=50, label_size=50),
            bottom_axis=fch.ChartAxis(title=ft.Text(x_name, size=10, weight="bold"), title_size=32, label_size=30),
            data_series=[
                fch.LineChartData(points=[fch.LineChartDataPoint(float(px), float(py)) for px, py in zip(x, y)], color=ft.Colors.GREEN_500, stroke_width=3),
                fch.LineChartData(points=[fch.LineChartDataPoint(float(px), float(py)) for px, py in zip(x_line, y_line)], color=ft.Colors.BLACK, stroke_width=2)
            ]
        )
        stats_view = ft.Row([ft.Text(f"R²={fmt_stat(stats['r2'])}", size=10, weight="bold"), ft.Text(f"F={fmt_stat(stats['f'])}", size=10, weight="bold"), ft.Text(f"n={stats['n']}", size=10, weight="bold")], wrap=True, spacing=10)
        return ft.Column([stats_view, chart], spacing=5)

    def make_crosstab_table(ct_percent):
        columns = [ft.DataColumn(ft.Text("항목"))] + [ft.DataColumn(ft.Text(str(col))) for col in ct_percent.columns]
        rows = [ft.DataRow(cells=[ft.DataCell(ft.Text(str(idx)))] + [ft.DataCell(ft.Text(f"{val:.1f}%")) for val in row]) for idx, row in ct_percent.iterrows()]
        return ft.Column([ft.Text("행 기준 비율(%)", size=14, weight="bold"), ft.DataTable(columns=columns, rows=rows)], scroll=ft.ScrollMode.AUTO)

    def show_main_screen():
        page.clean()
        page.scroll = None # 수정됨: ft.ScrollMode.NONE 대신 None 사용
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        
        btn_style = ft.ButtonStyle(
            text_style=ft.TextStyle(size=22, weight="bold"),
        )

        page.add(
            ft.Column(
                [
                    ft.Text("학생 자퇴 데이터 분석 시스템", 
                            size=42, 
                            weight="bold", 
                            color=ft.Colors.BLUE_GREY_800,
                            text_align=ft.TextAlign.CENTER),
                    ft.Container(height=60),
                    ft.FilledButton(
                        "자유 분석 모드 실행하기", 
                        icon=ft.Icons.ANALYTICS, 
                        on_click=start_app, 
                        width=450, 
                        height=85, 
                        style=btn_style
                    ),
                    ft.Container(height=30),
                    ft.OutlinedButton(
                        "대표 차트 5개 요약 보기", 
                        icon=ft.Icons.INSERT_CHART_OUTLINED, 
                        on_click=show_summary_charts, 
                        width=450, 
                        height=85, 
                        style=btn_style
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True
            )
        )
        page.update()

    def show_summary_charts(e):
        page.clean()
        page.scroll = ft.ScrollMode.AUTO
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.horizontal_alignment = ft.CrossAxisAlignment.START
        
        # ... (이하 동일한 로직)
        dropout_data = df_origin[df_origin['Dropout'] == 1]
        dropout_gender = dropout_data['Gender'].value_counts()
        total_dropout = len(dropout_data)
        chart1 = fch.PieChart(
            sections=[
                fch.PieChartSection(
                    value=float(v),
                    title=f"{k}\n{(v/total_dropout*100):.1f}%",
                    color=ft.Colors.BLUE_400 if k == "Male" else ft.Colors.RED_400,
                    radius=50,
                    title_style=ft.TextStyle(size=11, weight="bold", color=ft.Colors.WHITE)
                ) for k, v in dropout_gender.items()
            ],
            center_space_radius=40, expand=True
        )

        dept_dropout = df_origin.groupby('Department')['Dropout'].mean().sort_values(ascending=False)
        chart2 = make_bar_chart(dept_dropout, "학과", "자퇴율")

        df_origin['stress_bin'] = pd.cut(df_origin['Stress_Index'], bins=10, duplicates='drop')
        stress_agg = df_origin.groupby('stress_bin', observed=False)['Dropout'].mean()
        stress_stats_data = stress_agg.reset_index()
        stress_stats_data['x'] = stress_stats_data['stress_bin'].apply(lambda i: i.mid if pd.notnull(i) else 0)
        s_stats = calc_simple_regression_stats(stress_stats_data['x'], stress_stats_data['Dropout'])
        stress_agg.index = stress_agg.index.map(lambda x: round(x.mid, 1) if pd.notnull(x) else 0)
        chart3 = ft.Column([
            ft.Row([ft.Text(f"R²={fmt_stat(s_stats['r2'])}", size=10, weight="bold"), ft.Text(f"F={fmt_stat(s_stats['f'])}", size=10, weight="bold")], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
            make_bar_chart(stress_agg, "스트레스 지수", "자퇴율", bar_color=ft.Colors.RED_400)
        ], horizontal_alignment="center", spacing=5)

        df_origin['att_bin'] = pd.cut(df_origin['Attendance_Rate'], bins=10, duplicates='drop')
        att_dropout = df_origin.groupby('att_bin', observed=False)['Dropout'].mean().reset_index()
        att_dropout['x'] = att_dropout['att_bin'].apply(lambda i: i.mid if pd.notnull(i) else 0)
        chart4 = make_line_chart(att_dropout['x'], att_dropout['Dropout'], "출석률", "자퇴율")
        
        df_origin['study_bin'] = pd.cut(df_origin['Study_Hours_per_Day'], bins=15, duplicates='drop')
        study_gpa = df_origin.groupby('study_bin', observed=False)['GPA'].mean().reset_index()
        study_gpa['x'] = study_gpa['study_bin'].apply(lambda i: i.mid if pd.notnull(i) else 0)
        chart5 = make_line_chart(study_gpa['x'], study_gpa['GPA'], "하루 공부 시간", "평균 GPA")

        page.add(
            ft.Row([ft.Text("데이터 요약 보고서", size=25, weight="bold"), ft.Container(expand=True), ft.OutlinedButton("메인", on_click=lambda _: show_main_screen())]),
            ft.Divider(),
            ft.ResponsiveRow([
                ft.Container(ft.Column([ft.Text("자퇴생 성별 비율", weight="bold"), ft.Container(chart1, height=200)], horizontal_alignment="center"), col={"sm": 6}, padding=10, bgcolor="white", border_radius=15),
                ft.Container(ft.Column([ft.Text("학과별 자퇴율", weight="bold"), chart2], horizontal_alignment="center"), col={"sm": 6}, padding=10, bgcolor="white", border_radius=15),
                ft.Container(ft.Column([ft.Text("스트레스 vs 자퇴율", weight="bold"), chart3], horizontal_alignment="center"), col={"sm": 6}, padding=10, bgcolor="white", border_radius=15),
                ft.Container(ft.Column([ft.Text("출석률 vs 자퇴율", weight="bold"), chart4], horizontal_alignment="center"), col={"sm": 6}, padding=10, bgcolor="white", border_radius=15),
                ft.Container(ft.Column([ft.Text("공부 시간 vs GPA (회귀)", weight="bold"), chart5], horizontal_alignment="center"), col={"sm": 12}, padding=10, bgcolor="white", border_radius=15, height=420),
            ], spacing=20),
        )
        page.update()

    def start_app(e):
        page.clean()
        page.scroll = ft.ScrollMode.AUTO
        page.vertical_alignment = ft.MainAxisAlignment.START; page.horizontal_alignment = ft.CrossAxisAlignment.START
        # ... (이하 동일)
        df_clean = df_origin[[c for c in df_origin.columns if c not in EXCLUDE_COLS]].copy()
        selectable_cols = list(df_clean.columns)
        result_area = ft.Container()
        x_dropdown = ft.Dropdown(label="X 변수 선택", width=300, options=[ft.dropdown.Option(col) for col in selectable_cols], value=selectable_cols[0])
        y_dropdown = ft.Dropdown(label="Y 변수 선택", width=300, options=[ft.dropdown.Option(col) for col in selectable_cols], value=selectable_cols[1])

        def update_chart(e=None):
            x, y = x_dropdown.value, y_dropdown.value
            if x == y: result_area.content = ft.Text("서로 다른 변수를 선택하세요.", color="red", weight="bold"); page.update(); return
            sub = df_clean[[x, y]].dropna(); xk, yk = col_kind(x), col_kind(y)
            if xk == "categorical" and yk in {"numeric", "binary"}:
                content = make_bar_chart(sub.groupby(x)[y].mean(), x, y)
            elif xk in {"numeric", "binary"} and yk in {"numeric", "binary"}:
                if sub[x].nunique() > 12:
                    sub["x_bin"] = pd.cut(sub[x], bins=10, duplicates="drop")
                    agg = sub.groupby("x_bin", observed=False)[y].mean().reset_index().dropna()
                    agg["x_mid"] = agg["x_bin"].apply(lambda iv: (iv.left + iv.right) / 2)
                    content = make_line_chart(agg["x_mid"], agg[y], x, y)
                else:
                    agg = sub.groupby(x)[y].mean().reset_index().sort_values(x)
                    content = make_line_chart(agg[x], agg[y], x, y)
            elif xk in {"numeric", "binary"} and yk == "categorical":
                content = make_bar_chart(sub.groupby(y)[x].mean(), y, x)
            else:
                ct = pd.crosstab(sub[x], sub[y], normalize="index") * 100
                content = make_crosstab_table(ct)
            result_area.content = ft.Container(padding=20, border_radius=12, bgcolor="white", content=content); page.update()

        page.add(ft.Row([ft.Text("자유 분석 모드", size=25, weight="bold"), ft.Container(expand=True), ft.OutlinedButton("메인", on_click=lambda _: show_main_screen())]),
                 ft.Row([x_dropdown, y_dropdown, ft.FilledButton("분석하기", on_click=update_chart)]), result_area)
        update_chart()

    show_main_screen()

ft.run(main)