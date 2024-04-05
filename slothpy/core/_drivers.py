# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


def ensure_ready(func):
    """
    A decorator that checks if the instance's 'ready' attribute is True.
    If not, it calls the instance's 'run()' method before executing the
    decorated function.
    """
    def wrapper(self, *args, **kwargs):
        if not self._ready:
            self.run()
        return func(self, *args, **kwargs)
    
    return wrapper