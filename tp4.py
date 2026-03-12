import flet as ft
import flet_charts as fch
import pandas as pd
##

def main(page: ft.Page):

    page.title = "Student Dropout Analysis Dashboard"
    page.scroll = "auto"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    data_path = "/home/scienceodysseia/py_work/data/student_dropout_dataset_v3.csv"

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
                        width=18
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
        ),

        left_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(value=0.2, label=ft.Text("0.2")),
                fch.ChartAxisLabel(value=0.4, label=ft.Text("0.4")),
                fch.ChartAxisLabel(value=0.6, label=ft.Text("0.6")),
                fch.ChartAxisLabel(value=0.8, label=ft.Text("0.8")),
                fch.ChartAxisLabel(value=1.0, label=ft.Text("1.0")),
            ]
        )
    )

    # ---------------- 스트레스 vs 자퇴율 ----------------

    df["Stress_Int"] = df["Stress_Index"].round().astype(int)
    stress_rate = df.groupby("Stress_Int")["Dropout"].mean()

    stress_chart = fch.BarChart(

        groups=[
            fch.BarChartGroup(
                x=int(k),
                rods=[
                    fch.BarChartRod(
                        from_y=0,
                        to_y=float(v),
                        width=15
                    )
                ]
            )
            for k, v in stress_rate.items()
        ],

        max_y=1,

        bottom_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(
                    value=i,
                    label=ft.Text(str(i))
                )
                for i in range(1, 11)
            ]
        ),

        left_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(value=0.2, label=ft.Text("0.2")),
                fch.ChartAxisLabel(value=0.4, label=ft.Text("0.4")),
                fch.ChartAxisLabel(value=0.6, label=ft.Text("0.6")),
                fch.ChartAxisLabel(value=0.8, label=ft.Text("0.8")),
                fch.ChartAxisLabel(value=1.0, label=ft.Text("1.0")),
            ]
        )
    )

    # ---------------- 출석률 vs 자퇴율 ----------------

    df["Attend_Bin"] = (df["Attendance_Rate"] // 5) * 5
    attend_rate = df.groupby("Attend_Bin")["Dropout"].mean().reset_index()

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
                curved=True
            )
        ],

        bottom_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(
                    value=x,
                    label=ft.Text(f"{x}%")
                )
                for x in range(40, 101, 10)
            ]
        ),

        left_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(value=0.2, label=ft.Text("0.2")),
                fch.ChartAxisLabel(value=0.4, label=ft.Text("0.4")),
                fch.ChartAxisLabel(value=0.6, label=ft.Text("0.6")),
                fch.ChartAxisLabel(value=0.8, label=ft.Text("0.8")),
                fch.ChartAxisLabel(value=1.0, label=ft.Text("1.0")),
            ]
        )
    )

    # ---------------- 공부시간 vs GPA ----------------

    df_agg = df.groupby(df["Study_Hours_per_Day"].round(1))["GPA"].mean().reset_index()

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
                curved=True
            )
        ],

        bottom_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(value=i, label=ft.Text(f"{i}h"))
                for i in range(0, 10)
            ]
        ),

        left_axis=fch.ChartAxis(
            labels=[
                fch.ChartAxisLabel(value=0, label=ft.Text("0")),
                fch.ChartAxisLabel(value=1, label=ft.Text("1")),
                fch.ChartAxisLabel(value=2, label=ft.Text("2")),
                fch.ChartAxisLabel(value=3, label=ft.Text("3")),
                fch.ChartAxisLabel(value=4, label=ft.Text("4")),
            ]
        )
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
            card("출석률 vs 자퇴율", attend_chart)
        ]),

        card("공부시간 vs GPA", gpa_chart, 920)
    )


ft.run(main)