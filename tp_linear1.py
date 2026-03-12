import flet as ft
import flet_charts as fch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def main(page: ft.Page):

    page.title = "Student Dropout Analysis Dashboard"
    page.scroll = "auto"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    data_path = r"/home/scienceodysseia/py_work/tp/student_dropout_dataset_v3.csv"

    try:
        df = pd.read_csv(data_path)
    except:
        page.add(ft.Text("데이터 파일을 찾을 수 없습니다.", color="red"))
        return

    df = df.dropna(subset=[
        "Study_Hours_per_Day",
        "GPA",
        "Stress_Index",
        "Attendance_Rate"
    ])

    # ---------------- 성별 자퇴 비중 ----------------

    gender_dropout = df[df["Dropout"] == 1]["Gender"].value_counts()

    pie_chart = fch.PieChart(
        sections=[
            fch.PieChartSection(
                value=float(v),
                title=f"{k}\n{v}",
                radius=80
            )
            for k, v in gender_dropout.items()
        ],
        center_space_radius=40
    )

    # ---------------- 학과별 자퇴율 ----------------

    dept_rate = df.groupby("Department")["Dropout"].mean()

    dept_chart = fch.BarChart(

        groups=[
            fch.BarChartGroup(
                x=i,
                rods=[
                    fch.BarChartRod(
                        from_y=0,
                        to_y=float(v),
                        width=18,
                        color=ft.Colors.BLUE_400
                    )
                ]
            )
            for i, v in enumerate(dept_rate.values)
        ],

        max_y=1,

        bottom_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(
                    value=i,
                    label=ft.Text(n, size=9)
                )
                for i, n in enumerate(dept_rate.index)
            ]
        )
    )

    # ---------------- 스트레스 vs 자퇴율 ----------------

    df["Stress_Int"] = df["Stress_Index"].round().astype(int)
    stress_rate = df.groupby("Stress_Int")["Dropout"].mean()

    X = stress_rate.index.values.reshape(-1, 1)
    y = stress_rate.values

    model = LinearRegression()
    model.fit(X, y)

    x_line = np.linspace(X.min(), X.max(), 50)
    y_line = model.predict(x_line.reshape(-1, 1))

    stress_bar = fch.BarChart(

        groups=[
            fch.BarChartGroup(
                x=i,
                rods=[
                    fch.BarChartRod(
                        from_y=0,
                        to_y=float(v),
                        width=18,
                        color=ft.Colors.ORANGE_400
                    )
                ]
            )
            for i, v in enumerate(stress_rate.values)
        ],

        max_y=1,

        bottom_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(value=i, label=ft.Text(str(k)))
                for i, k in enumerate(stress_rate.index)
            ]
        )
    )

    stress_line = fch.LineChart(

        min_x=0,
        max_x=len(stress_rate)-1,
        min_y=0,
        max_y=1,

        data_series=[
            fch.LineChartData(
                points=[
                    fch.LineChartDataPoint(
                        float((x - X.min())/(X.max()-X.min()) * (len(stress_rate)-1)),
                        float(y)
                    )
                    for x, y in zip(x_line, y_line)
                ],
                curved=False,
                color=ft.Colors.BLACK,
                stroke_width=3
            )
        ]
    )

    stress_chart = ft.Stack([
        stress_bar,
        stress_line
    ])

    # ---------------- 출석률 vs 자퇴율 ----------------

    df["Attend_Bin"] = (df["Attendance_Rate"] // 5) * 5
    attend_rate = df.groupby("Attend_Bin")["Dropout"].mean().reset_index()

    X = attend_rate["Attend_Bin"].values.reshape(-1, 1)
    y = attend_rate["Dropout"].values

    model = LinearRegression()
    model.fit(X, y)

    x_line = np.linspace(X.min(), X.max(), 50)
    y_line = model.predict(x_line.reshape(-1, 1))

    attend_chart = fch.LineChart(

        min_x=35,
        max_x=100,
        min_y=0,
        max_y=1,

        data_series=[

            fch.LineChartData(
                points=[
                    fch.LineChartDataPoint(
                        float(row["Attend_Bin"]),
                        float(row["Dropout"])
                    )
                    for _, row in attend_rate.iterrows()
                ],
                curved=True,
                color=ft.Colors.GREEN_400
            ),

            fch.LineChartData(
                points=[
                    fch.LineChartDataPoint(float(x), float(y))
                    for x, y in zip(x_line, y_line)
                ],
                curved=False,
                color=ft.Colors.BLACK
            )
        ]
    )

    # ---------------- 공부시간 vs GPA ----------------

    df_agg = df.groupby(df["Study_Hours_per_Day"].round(1))["GPA"].mean().reset_index()

    X = df_agg["Study_Hours_per_Day"].values.reshape(-1, 1)
    y = df_agg["GPA"].values

    model = LinearRegression()
    model.fit(X, y)

    x_line = np.linspace(X.min(), X.max(), 50)
    y_line = model.predict(x_line.reshape(-1, 1))

    gpa_chart = fch.LineChart(

        min_x=0,
        max_x=9,
        min_y=0,
        max_y=4.5,

        data_series=[

            fch.LineChartData(
                points=[
                    fch.LineChartDataPoint(
                        float(r["Study_Hours_per_Day"]),
                        float(r["GPA"])
                    )
                    for _, r in df_agg.iterrows()
                ],
                curved=True,
                color=ft.Colors.PURPLE_400
            ),

            fch.LineChartData(
                points=[
                    fch.LineChartDataPoint(float(x), float(y))
                    for x, y in zip(x_line, y_line)
                ],
                curved=False,
                color=ft.Colors.BLACK
            )
        ]
    )

    # ---------------- 카드 UI ----------------

    def card(title, chart, width=450):

        return ft.Container(
            width=width,
            height=420,
            padding=20,
            border_radius=10,
            bgcolor="white",
            content=ft.Column([
                ft.Text(title, size=18, weight="bold"),
                chart
            ])
        )

    page.add(

        ft.Text(
            "Student Dropout Analysis Dashboard",
            size=30,
            weight="bold"
        ),

        ft.Row([
            card("자퇴생 성별 비중", pie_chart),
            card("학과별 자퇴율", dept_chart)
        ]),

        ft.Row([
            card("스트레스 vs 자퇴율", stress_chart),
            card("출석률 vs 자퇴율 (Regression)", attend_chart)
        ]),

        card("공부시간 vs GPA (Regression)", gpa_chart, 920)
    )


ft.run(main)