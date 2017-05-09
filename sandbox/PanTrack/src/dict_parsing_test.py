# Dictionary Parsing Test
# https://www.github.com/mvoellmy/

write_dict = {}
write_dict['string'] = 'string'
write_dict['int'] = 1
write_dict['float'] = 3.2
write_dict['tuple'] = (1, 2)
write_dict['list'] = [1, 2, 5]
# ...add additional datatypes here to see if it works...
# write_dict['data_type'] = data

print('Dictionary Parsing Test')

# write dictionary to text file
with open('dict_test.txt', 'w') as csvfile:
    csvfile.write(repr(write_dict))

# read dictionary from text file
with open('dict_test.txt', 'r') as csvfile:
    read_dict = eval(csvfile.read())

# if the outputs of write and read dict are the same it worked
print('#############')
print('write:\t{}'.format(type(write_dict)))
for key, value in write_dict.items():
    print('{0}:\t{2}\t{1}'.format(key, value, type(value)))

print('#############')
print('read\t{}'.format(type(read_dict)))
for key, value in read_dict.items():
    print('{0}:\t{2}\t{1}'.format(key, value, type(value)))
