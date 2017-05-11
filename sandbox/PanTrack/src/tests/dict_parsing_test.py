# Dictionary Parsing Test
# https://www.github.com/mvoellmy/
import numpy as np

write_dict = {}
write_dict['string'] = 'string'
write_dict['int'] = 1
write_dict['float'] = 3.2
write_dict['tuple'] = (1, 2)
write_dict['list'] = [1, 2, 5]
write_dict['dict'] = {'tuple': (1, 2),
                      'string': 'this is a string'}
# ...add additional datatypes here to see if it works...
# These DO NOT WORK:
# write_dict['np_array'] = np.zeros(3)

print('Dictionary Parsing Test')

# write dictionary to text file
with open('dict_test.txt', 'w') as file:
    file.write(repr(write_dict))

# read dictionary from text file
with open('dict_test.txt', 'r') as file:
    read_dict = eval(file.read())

# if the outputs of write and read dict are the same it worked
print('#############')
print('write:\t{}'.format(type(write_dict)))
for key, value in write_dict.items():
    print('{0}:\t{2}\t{1}'.format(key, value, type(value)))

print('#############')
print('read\t{}'.format(type(read_dict)))
for key, value in read_dict.items():
    print('{0}:\t{2}\t{1}'.format(key, value, type(value)))

