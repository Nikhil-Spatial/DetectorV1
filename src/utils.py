def get_labels(groups, filename):
    group = groups.get_group(filename).drop(columns=["filename"])

    objects = []
    for i in range(len(group)):
        obj_tuple = tuple(group.iloc[i])
        objects.append(obj_tuple)

    return objects
