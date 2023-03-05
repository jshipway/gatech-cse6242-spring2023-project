# Imports
import pymap3d as pm3


_MILES_2_METERS = 1609.344
_MILES_2_NMILES = 0.868976
_MILES_2_FEET = 5280.


def distance_conversion(val, orig_unit, new_unit):
    """
    Converts a given value from its original units to its new unit.

    :param val: float, the value to convert the units of
    :param orig_unit: str, the original unit type, must be one of ['miles', 'feet', 'meters', 'nautical miles']
    :param new_unit: str, the new unit type, must be one of ['miles', 'feet', 'meters', 'nautical miles']
    :return: the value converted to the new unit
    """
    _allowed_units = ['miles', 'feet', 'meters', 'nauticalmiles']
    if orig_unit.lower().replace(' ', '') not in _allowed_units:
        raise ValueError('The original unit provided is not an accepted unit type. Received {}'.format(orig_unit))
    if new_unit.lower().replace(' ', '') not in _allowed_units:
        raise ValueError('The new unit provided is not an accepted unit type. Received {}'.format(new_unit))

    orig_unit = orig_unit.lower().replace(' ', '')
    new_unit = new_unit.lower().replace(' ', '')

    # NOTE: This could be handled more eloquently as a switch case
    # But would require us agreeing to develop in Python version 3.10.x
    if orig_unit == new_unit:  # Handle case where no conversion is needed
        return val
    elif (orig_unit == 'miles') and (new_unit == 'meters'):  # Block of "miles" as origin unit
        return val * _MILES_2_METERS
    elif (orig_unit == 'miles') and (new_unit == 'feet'):
        return val * _MILES_2_FEET
    elif (orig_unit == 'miles') and (new_unit == 'nauticalmiles'):
        return val * _MILES_2_NMILES
    elif (orig_unit == 'meters') and (new_unit == 'feet'):  # Block of "meters" as origin unit
        return val * _MILES_2_FEET / _MILES_2_METERS
    elif (orig_unit == 'meters') and (new_unit == 'nauticalmiles'):
        return val * _MILES_2_NMILES / _MILES_2_METERS
    elif (orig_unit == 'meters') and (new_unit == 'miles'):
        return val / _MILES_2_METERS
    elif (orig_unit == 'feet') and (new_unit == 'meters'):  # Block of "feet" as origin unit
        return val * _MILES_2_METERS / _MILES_2_FEET
    elif (orig_unit == 'feet') and (new_unit == 'nauticalmiles'):
        return val * _MILES_2_NMILES / _MILES_2_FEET
    elif (orig_unit == 'feet') and (new_unit == 'miles'):
        return val / _MILES_2_FEET
    elif (orig_unit == 'nauticalmiles') and (new_unit == 'meters'):  # Block of "nautical miles" as origin unit
        return val * _MILES_2_METERS / _MILES_2_NMILES
    elif (orig_unit == 'nauticalmiles') and (new_unit == 'feet'):
        return val * _MILES_2_FEET / _MILES_2_NMILES
    elif (orig_unit == 'nauticalmiles') and (new_unit == 'miles'):
        return val / _MILES_2_NMILES


def formatted_latlon_2_decimal(lat, lon):
    """
    Converts a string formatted latitude/longitude coordinate into a decimal formatted coordinate.

    :param lat: str, latitude coordinate formatted (e.g. "N 39 7' 30.0"" or "N39:07:30.0")
    :param lon: str, longitude coordinate formatted (e.g. "W 94 30' 0.0"" or "W94:30:00.0")
    :return: the latitude and longitude formatted as decimals
    """

    if not isinstance(lat, str):
        raise TypeError('Expected latitude to be a string, received {}'.format(type(lat)))

    if not isinstance(lon, str):
        raise TypeError('Expected longitude to be a string, received {}'.format(type(lon)))

    lat_dir = lat[0].upper()
    lon_dir = lon[0].upper()

    if lat_dir not in ['N', 'S']:
        raise ValueError('Expected a cardinal direction coordinate at the beginning of the formatted '
                         'latitude coordinate. Received {}'.format(lat_dir))
    if lon_dir not in ['E', 'W']:
        raise ValueError('Expected a cardinal direction coordinate at the beginning of the formatted '
                         'longitude coordinate. Received {}'.format(lon_dir))

    if ':' in lat:
        lat_parse = lat.replace(' ', '').replace(lat_dir, '').split(':')
    else:
        lat_parse = lat.replace(lat_dir, '').split(' ')
        if lat_parse[0] == '':
            lat_parse = lat_parse[1:]

    if ':' in lon:
        lon_parse = lon.replace(' ', '').replace(lon_dir, '').split(':')
    else:
        lon_parse = lon.replace(lon_dir, '').split(' ')
        if lon_parse[0] == '':
            lon_parse = lon_parse[1:]

    if len(lat_parse) != 3:
        raise ValueError('Expected to find three parts to the latitude coordinate, found {}. '
                         'Split looks as follows: {}'.format(len(lat_parse), lat_parse))
    if len(lon_parse) != 3:
        raise ValueError('Expected to find three parts to the longitude coordinate, found {}. '
                         'Split looks as follows: {}'.format(len(lon_parse), lon_parse))

    lat_dec = int(lat_parse[0]) + (int(lat_parse[1].replace("'", '')) / 60.) + \
        (float(lat_parse[2].replace('"', '')) / 60. / 60.)
    lat_dec = -1 * lat_dec if lat_dir == 'S' else lat_dec  # Multiply by -1 if the direction is south rather than north

    lon_dec = int(lon_parse[0]) + (int(lon_parse[1].replace("'", '')) / 60.) + \
        (float(lon_parse[2].replace('"', '')) / 60. / 60.)
    lon_dec = -1 * lon_dec if lon_dir == 'W' else lon_dec  # Multiply by -1 if the direction is west rather than east

    return lat_dec, lon_dec


def decimal_latlon_2_formatted(lat, lon):
    """
    Converts a decimal latitude/longitude coordinate into a string formatted coordinate.

    :param lat: float, latitude coordinate as a decimal (e.g. 39.125)
    :param lon: float, longitude coordinate as a decimal (e.g. -94.5)
    :return: the latitude and longitude formatted as strings
    """

    if not isinstance(lat, float):
        raise TypeError('Expected latitude to be a float, received {}'.format(type(lat)))
    if not isinstance(lon, float):
        raise TypeError('Expected longitude to be a float, received {}'.format(type(lon)))

    lat_dir = 'N' if lat > 0 else 'S'
    lon_dir = 'E' if lon > 0 else 'W'
    lat = abs(lat)
    lon = abs(lon)

    lat_deg = int(lat // 1)
    lon_deg = int(lon // 1)
    lat -= lat_deg
    lon -= lon_deg
    lat *= 60.
    lon *= 60.
    lat_min = int(lat // 1)
    lon_min = int(lon // 1)
    lat -= lat_min
    lon -= lon_min
    lat *= 60.
    lon *= 60.
    lat_sec = round(float(lat), 4)
    lon_sec = round(float(lon), 4)

    lat_format = '{}{}:{}:{}'.format(lat_dir, lat_deg, lat_min, lat_sec)
    lon_format = '{}{}:{}:{}'.format(lon_dir, lon_deg, lon_min, lon_sec)

    return lat_format, lon_format


def get_direction_range(lat_lon0, lat_lon1):
    """
    Gets the direction from the first lat/long coordinate to the second, as well as
    the straight-line distance between the two.

    :param lat_lon0: tuple or list-like, two elements representing the latitude and longitude of the first coordinate
        (e.g. [39.125, -94.5] is approximately Kansas City, MO)
    :param lat_lon1: tuple or list-like, two elements representing the latitude and longitude of the second coordinate
    :return: azimuth (direction) and range in miles from coordinate 1 to coordinate 2
    """
    try:
        lat_lon0 = list(lat_lon0)
        lat_lon1 = list(lat_lon1)
    except Exception as _:
        raise ValueError('Could not type-cast the lat/long coordinates to lists. Ensure the inputs are list-like.')

    if not len(lat_lon0) == 2:
        raise ValueError('The first lat/long coordinate did not contain two elements, contained {}'
                         .format(len(lat_lon0)))

    if not len(lat_lon1) == 2:
        raise ValueError('The second lat/long coordinate did not contain two elements, contained {}'
                         .format(len(lat_lon1)))

    az, _, rng = pm3.geodetic2aer(lat_lon1[0], lat_lon1[1], 0, lat_lon0[0], lat_lon0[1], 0)
    rng_corrected = distance_conversion(rng, 'meters', 'miles')

    return az, rng_corrected


if __name__ == '__main__':
    # The following are a few examples of how to run these functions
    LAT0_FORMAT = '''N 40 51' 39.4612"'''
    LON0_FORMAT = '''W 95 16' 04.1263"'''

    LAT1_FORMAT = 'N46:08:19.2715'
    LON1_FORMAT = 'W102:42:34.7102'

    print('Lat/Long Coordinate 0: {}, {}'.format(LAT0_FORMAT, LON0_FORMAT))
    print('Lat/Long Coordinate 1: {}, {}'.format(LAT1_FORMAT, LON1_FORMAT))

    LATLON0_DEC = formatted_latlon_2_decimal(LAT0_FORMAT, LON0_FORMAT)
    LATLON1_DEC = formatted_latlon_2_decimal(LAT1_FORMAT, LON1_FORMAT)

    print('Lat/Long Coordinate 0 (decimal): {}, {}'.format(LATLON0_DEC[0], LATLON0_DEC[1]))
    print('Lat/Long Coordinate 1 (decimal): {}, {}'.format(LATLON1_DEC[0], LATLON1_DEC[1]))

    DIRECTION, MILES = get_direction_range(LATLON0_DEC, LATLON1_DEC)
    print('Direction from coordinate 0 to coordinate 1: {}'.format(round(DIRECTION, 4)))
    print('Miles from coordinate 0 to coordinate 1: {}'.format(round(MILES, 4)))

    FEET = distance_conversion(MILES, 'miles', 'feet')
    print('Feed from coordinate 0 to coordinate 1: {}'.format(round(FEET, 4)))

    LAT0_REFORMAT, LON0_REFORMAT = decimal_latlon_2_formatted(LATLON0_DEC[0], LATLON0_DEC[1])
    print('Lat/Long Coordinate 1 (converted from decimal back to formatted): {}, {}'
          .format(LAT0_REFORMAT, LON0_REFORMAT))
