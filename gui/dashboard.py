from PyQt6.QtCharts import QBarSet
from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox


def update_dashboard(chart, series, axis_x, axis_y, table, tracker):
    """Atualiza gráfico de barras e tabela do dashboard."""
    series.clear()
    exs, errs = [], []
    for ex in tracker.total.keys():
        st = tracker.stats(ex)
        tot_err = sum(v["c"] for v in st.values())
        exs.append(ex)
        errs.append(tot_err)

    if errs:
        bar_set = QBarSet("Erros")
        for e in errs:
            bar_set.append(e)
        series.append(bar_set)
        axis_x.clear()
        axis_x.append(exs)
        axis_y.setRange(0, max(errs) + 1)

    # ---------- tabela ----------
    table.setRowCount(0)
    for ex, tot in tracker.total.items():
        st = tracker.stats(ex)
        terr = sum(v["c"] for v in st.values())
        pct = (terr / tot * 100) if tot else 0.0
        row = table.rowCount()
        table.insertRow(row)
        for i, txt in enumerate([ex, str(tot), str(terr), f"{pct:.1f}%"]):
            table.setItem(row, i, QTableWidgetItem(txt))


def reset_data_action(parent, tracker, rep_state_factory):
    """Pergunta e apaga JSON + reinicia estados."""
    reply = QMessageBox.question(
        parent,
        "Confirmar Reset",
        "Tem certeza que deseja apagar todos os dados?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if reply == QMessageBox.StandardButton.Yes:
        tracker.file_reset()
        # reinicia máquinas de rep
        rep_state_factory.clear()
        return True
    return False