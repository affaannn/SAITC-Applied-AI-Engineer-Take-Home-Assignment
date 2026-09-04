from datetime import date


def calculate_partial_year_leave(
    start_date: date,
    end_date: date,
    annual_entitlement: float = 24.0,
) -> dict:

    monthly_rate = annual_entitlement / 12

    months_counted = 0
    details = []

    year = start_date.year
    month = start_date.month

    while (year, month) <= (
        end_date.year,
        end_date.month,
    ):

        first_month = (
            year == start_date.year
            and month == start_date.month
        )

        last_month = (
            year == end_date.year
            and month == end_date.month
        )

        if first_month and last_month:
            days_served = (
                end_date - start_date
            ).days + 1

            counts = days_served >= 15

        elif first_month:
            # Number of calendar days served in joining month.
            import calendar

            days_in_month = calendar.monthrange(
                year,
                month,
            )[1]

            days_served = (
                days_in_month
                - start_date.day
                + 1
            )

            counts = days_served >= 15

        elif last_month:
            days_served = end_date.day
            counts = days_served >= 15

        else:
            days_served = None
            counts = True

        if counts:
            months_counted += 1

        details.append(
            {
                "year": year,
                "month": month,
                "days_served": days_served,
                "counts": counts,
            }
        )

        month += 1

        if month == 13:
            month = 1
            year += 1

    entitlement = (
        months_counted
        * monthly_rate
    )

    return {
        "months_counted": months_counted,
        "monthly_rate": monthly_rate,
        "entitlement": round(
            entitlement,
            2,
        ),
        "details": details,
    }