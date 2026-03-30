AttributeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/correlation-analysis-between-mechanical-properties-and-customer-quality-grading/app.py", line 121, in <module>
    sns.histplot(
    ~~~~~~~~~~~~^
        data=temp_df,
        ^^^^^^^^^^^^^
    ...<9 lines>...
        warn_singular=False
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/seaborn/distributions.py", line 1416, in histplot
    p.plot_univariate_histogram(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        multiple=multiple,
        ^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/seaborn/distributions.py", line 571, in plot_univariate_histogram
    artists = plot_func(
        hist["edges"],
    ...<4 lines>...
        **artist_kws,
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/matplotlib/__init__.py", line 1524, in inner
    return func(
        ax,
        *map(cbook.sanitize_sequence, args),
        **{k: cbook.sanitize_sequence(v) for k, v in kwargs.items()})
File "/home/adminuser/venv/lib/python3.14/site-packages/matplotlib/axes/_axes.py", line 2654, in bar
    r._internal_update(kwargs)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/matplotlib/artist.py", line 1233, in _internal_update
    return self._update_props(
           ~~~~~~~~~~~~~~~~~~^
        kwargs, "{cls.__name__}.set() got an unexpected keyword argument "
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        "{prop_name!r}")
        ^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/matplotlib/artist.py", line 1206, in _update_props
    raise AttributeError(
        errfmt.format(cls=type(self), prop_name=k),
        name=k)
