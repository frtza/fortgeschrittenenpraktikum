# Protokolle zum Fortgeschrittenenpraktikum für Physikstudierende

**Beschreibung**

Aus [Moodle](https://moodle.tu-dortmund.de/):
> Das Physikalische Praktikum für Fortgeschrittene ist eine einsemestrige Veranstaltung und vertieft die in den Grundpraktika erlernten Techniken.

Die Struktur dieses Projekts und die grundlegende Methodik sind an den
[Toolbox-Workshop](https://toolbox.pep-dortmund.org/notes/) von
[PeP et al. e.V.](https://pep-dortmund.org/) angelehnt. Als Hilfe stellt die
[Fachschaft](https://fachschaft-physik.tu-dortmund.de/wordpress/studium/praktikum/altprotokolle-fp/)
Altprotokolle zur Verfügung.

**Autoren**

Fritz Agildere ([fritz.agildere@udo.edu](mailto:fritz.agildere@udo.edu)) und
Amelie Strathmann ([amelie.strathmann@udo.edu](mailto:amelie.strathmann@udo.edu))

**Struktur**

Die Protokolle werden mit `make` als PDF-Datei ausgegeben. Im Hauptverzeichnis wird die allgemeine Konfiguration
vorgenommen. Die Unterverzeichnisse übernehmen diese standardmäßig. Die einzelnen Versuche enthalten wiederum die
Verzeichnisse `build`, in dem sich alle generierten Dateien befinden, und `content`, das der Struktur des Protokolls
entspricht:

1. Zielsetzung
2. Theorie
3. Durchführung
4. Auswertung
5. Diskussion

Zur graphischen Darstellung und um abgeleitete Messwerte automatisch zu berechnen, werden `python` Skripte
mit den entsprechenden Bibliotheken genutzt. Die Dokumente werden unter Anwendung von `lualatex` kompiliert.

Das Projekt *Fortgeschrittenenpraktikum* ist mit GNU/Linux kompatibel.
