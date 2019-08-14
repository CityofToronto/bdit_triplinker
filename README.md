# Triplinker

Triplinker is a Python package to link vehicle-for-hire trips together into
driver work shifts, allowing statistics of the driver population to be
estimated from trip origin-destination (OD) records. It is authored and
maintained by the [Big Data Innovation Team](https://www.toronto.ca/services-payments/streets-parking-transportation/road-safety/big-data-innovation-team/) at the
City of Toronto.

While Triplinker can be used on any set of origin-destination data, it was
built for analyzing OD data from ridesourcing platforms like Uber and Lyft, as
part of a [technical report](
https://www.toronto.ca/wp-content/uploads/2019/06/96c7-Report_v1.0_2019-06-21.pdf)
for the the City's [Vehicle-for-Hire bylaw review](
http://app.toronto.ca/tmmis/viewAgendaItemHistory.do?item=2019.GL6.31).
Consequently, it assumes a variable supply of active vehicles rather than a
fixed fleet of vehicles operated by drivers with regular work shifts (as with
eg. taxi companies).

## Usage

### Dependencies

```
networkx==2.2
numpy>=1.14
pandas>=0.23
pytest>=3.8.0
scikit-learn>=0.20
```

### Installation

To import `triplinker`, add this folder to the Python PATH, eg. with

```
import sys
sys.path.append('<FULL PATH OF bdit_triplinker>')
```

### Testing

To test the package, run the following in this folder:

```
pytest -s -v --pyargs triplinker
```

## Project Status

## License

Triplinker is licensed under the GNU General Public License v3.0 - see the
`LICENSE` file.
